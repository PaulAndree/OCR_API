import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import shutil
import uuid
import os
import tempfile
import time
from main_ocr import ocr_model, unload_models
import structlog
import logging
from typing import Optional


IDLE_UNLOAD_SECONDS = int(os.getenv("OCR_IDLE_UNLOAD_SECONDS", "3600"))
QUEUE_MAXSIZE = int(os.getenv("OCR_QUEUE_MAXSIZE", "2"))
QUEUE_RETRY_AFTER_SECONDS = int(os.getenv("OCR_QUEUE_RETRY_AFTER_SECONDS", "30"))

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True
)
logger = structlog.get_logger(service="ocr")


@dataclass
class OCRJob:
    temp_path: str
    output_path: str
    workspace: str
    file_name: str
    source_name: Optional[str]
    type_name: str
    doc_id: Optional[str]
    queued_at: float
    result: asyncio.Future


def run_ocr(job: OCRJob):
    ocr = ocr_model(job.output_path)
    if job.type_name == "pdf":
        ocr.get_pdf_text(job.temp_path, job.file_name)
    else:
        ocr.get_doc_text(job.temp_path, job.file_name)
    return ocr.as_dict()


async def ocr_worker(app: FastAPI):
    while True:
        job = await app.state.ocr_queue.get()
        if job is None:
            app.state.ocr_queue.task_done()
            return

        try:
            if job.result.cancelled():
                continue
            queue_wait_ms = int((time.monotonic() - job.queued_at) * 1000)
            started_at = time.monotonic()
            async with app.state.ocr_lock:
                app.state.ocr_busy = True
                app.state.last_activity = time.monotonic()
                logger.info(
                    event="ocr_start", type_name=job.type_name,
                    file_name=job.file_name, source_name=job.source_name,
                    doc_id=job.doc_id, queue_wait_ms=queue_wait_ms,
                    queue_depth=app.state.ocr_queue.qsize()
                )
                response = await asyncio.to_thread(run_ocr, job)

            processing_ms = int((time.monotonic() - started_at) * 1000)
            logger.info(
                event="ocr_done", type_name=job.type_name,
                file_name=job.file_name, source_name=job.source_name,
                doc_id=job.doc_id, queue_wait_ms=queue_wait_ms,
                processing_ms=processing_ms,
                queue_depth=app.state.ocr_queue.qsize()
            )
            if not job.result.done():
                job.result.set_result(response)
        except Exception as exc:
            logger.error(
                event="ocr_error", error=str(exc), type_name=job.type_name,
                file_name=job.file_name, source_name=job.source_name,
                doc_id=job.doc_id
            )
            if not job.result.done():
                job.result.set_exception(exc)
        finally:
            app.state.ocr_busy = False
            app.state.last_activity = time.monotonic()
            await asyncio.to_thread(shutil.rmtree, job.workspace, True)
            app.state.ocr_queue.task_done()


async def unload_when_idle(app: FastAPI):
    while True:
        await asyncio.sleep(60)
        if app.state.ocr_busy or not app.state.ocr_queue.empty():
            continue
        if time.monotonic() - app.state.last_activity < IDLE_UNLOAD_SECONDS:
            continue

        async with app.state.ocr_lock:
            idle = time.monotonic() - app.state.last_activity
            if app.state.ocr_queue.empty() and idle >= IDLE_UNLOAD_SECONDS:
                if await asyncio.to_thread(unload_models):
                    logger.info(event="ocr_models_unloaded", idle_s=int(idle))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    app.state.ocr_lock = asyncio.Lock()
    app.state.ocr_busy = False
    app.state.accepting_jobs = True
    app.state.last_activity = time.monotonic()
    worker = asyncio.create_task(ocr_worker(app))
    watcher = asyncio.create_task(unload_when_idle(app))
    logger.info(event="ocr_api_started", queue_capacity=QUEUE_MAXSIZE)
    yield
    app.state.accepting_jobs = False
    watcher.cancel()
    with suppress(asyncio.CancelledError):
        await watcher
    await app.state.ocr_queue.put(None)
    await worker
    logger.info(event="ocr_api_stopped")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ocr_busy": app.state.ocr_busy,
        "queue_depth": app.state.ocr_queue.qsize(),
        "queue_capacity": QUEUE_MAXSIZE,
    }


def copy_upload(source, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(source, buffer)


def queue_full_error():
    return HTTPException(
        status_code=503,
        detail="OCR queue is full",
        headers={"Retry-After": str(QUEUE_RETRY_AFTER_SECONDS)},
    )


@app.post("/services/ocr")
async def process_(
    file: UploadFile = File(...),
    type_name: str = Form(...),
    doc_id: Optional[str] = Form(None)
):
    normalized_type = type_name.strip().lower()
    if normalized_type not in {"pdf", "docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"File type ({type_name}) not supported"
        )
    if not app.state.accepting_jobs:
        raise HTTPException(status_code=503, detail="OCR service is shutting down")
    if app.state.ocr_queue.full():
        raise queue_full_error()

    workspace = tempfile.mkdtemp(prefix="ocr-")
    file_name = str(uuid.uuid4())
    temp_path = os.path.join(workspace, f"{file_name}.{normalized_type}")
    output_path = os.path.join(workspace, "output")
    try:
        await asyncio.to_thread(copy_upload, file.file, temp_path)
    except Exception:
        await asyncio.to_thread(shutil.rmtree, workspace, True)
        raise HTTPException(status_code=503, detail="Error saving uploaded file")
    finally:
        await file.close()

    result = asyncio.get_running_loop().create_future()
    job = OCRJob(
        temp_path=temp_path,
        output_path=output_path,
        workspace=workspace,
        file_name=file_name,
        source_name=file.filename,
        type_name=normalized_type,
        doc_id=doc_id,
        queued_at=time.monotonic(),
        result=result,
    )

    try:
        app.state.ocr_queue.put_nowait(job)
        app.state.last_activity = time.monotonic()
    except asyncio.QueueFull:
        await asyncio.to_thread(shutil.rmtree, workspace, True)
        raise queue_full_error()

    try:
        return await asyncio.shield(result)
    except asyncio.CancelledError:
        result.cancel()
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="OCR processing failed")
