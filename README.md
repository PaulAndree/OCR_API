# IARAG OCR Service

FastAPI service that extracts Markdown text from PDF and DOCX documents using the bundled [MinerU](https://github.com/opendatalab/MinerU) pipeline.

The API accepts one document per request, detects whether the document is Spanish or another language, runs layout-aware OCR, and returns the extracted Markdown. Work is serialized through a bounded in-process queue to avoid loading or running multiple copies of the OCR models on the same GPU.

## Features

- PDF text extraction and OCR with MinerU's automatic parse mode.
- DOCX support through headless LibreOffice conversion to PDF.
- Layout-aware Markdown generation with formula recognition enabled.
- Automatic Spanish detection (`latin` OCR model); all other inputs use the English model.
- Removal of image references, table-of-contents blocks, and trailing references sections from returned Markdown.
- A bounded request queue with `503` backpressure and a configurable `Retry-After` header.
- Lazy model loading and automatic model unloading after an idle period to release GPU memory.
- JSON structured request logs and a health endpoint.

## API

The application listens on port `8084`. The supplied Docker Compose configuration publishes it only on the host loopback interface at `http://127.0.0.1:8084`.

Interactive OpenAPI documentation is available at `/docs` while the service is running.

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "ocr_busy": false,
  "queue_depth": 0,
  "queue_capacity": 2
}
```

`queue_depth` reports jobs waiting in the queue; the currently running job is represented by `ocr_busy` and is not included in that count.

### Extract a document

```http
POST /services/ocr
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | file | yes | PDF or DOCX document to process. |
| `type_name` | string | yes | Must be `pdf` or `docx` (case-insensitive). The service uses this value rather than inferring the type from the filename. |
| `doc_id` | string | no | Identifier included in structured logs for request tracing; it is not returned. |

PDF example:

```bash
curl -X POST http://127.0.0.1:8084/services/ocr \
  -F 'file=@/path/to/document.pdf' \
  -F 'type_name=pdf' \
  -F 'doc_id=example-001'
```

DOCX example:

```bash
curl -X POST http://127.0.0.1:8084/services/ocr \
  -F 'file=@/path/to/document.docx' \
  -F 'type_name=docx' \
  -F 'doc_id=example-002'
```

Example response:

```json
{
  "ocr_language": "en",
  "content_parsed": "# Document title\n\nExtracted document text..."
}
```

`ocr_language` is `latin` when Spanish is detected from the first page and `en` otherwise. This is the model language passed to MinerU, not an ISO language code.

### Errors and backpressure

| Status | Meaning |
| --- | --- |
| `400` | `type_name` is not `pdf` or `docx`. |
| `422` | A required multipart field is missing or malformed. |
| `503` | The queue is full, the service is shutting down, the upload could not be saved, or OCR processing failed. |

When the queue is full, the response includes a `Retry-After` header. Its value is controlled by `OCR_QUEUE_RETRY_AFTER_SECONDS`.

## Run with Docker Compose

Requirements:

- Docker with the Compose plugin.
- An NVIDIA GPU and the NVIDIA Container Toolkit.
- An existing external Docker network named `n8n-net`.
- Internet access on the first OCR request unless MinerU models are already cached or configured locally.

Create the shared network once if it does not already exist:

```bash
docker network create n8n-net
```

Build and start the service:

```bash
cd /srv/iarag/services/ocr
docker compose up -d --build
docker compose logs -f ocr
```

Verify it from the host:

```bash
curl http://127.0.0.1:8084/health
```

Other containers attached to `n8n-net` can call `http://ocr:8084`. The published host port is intentionally bound to `127.0.0.1`, so it is not directly exposed on external host interfaces.

Stop the service with:

```bash
docker compose down
```

The image starts Uvicorn with one worker. Keep a single worker unless the deployment has enough GPU memory for an independent model cache and job queue in every worker.

## Run locally

Python 3.12 is used by the container. The package supports Python 3.10 through 3.13.

On Debian or Ubuntu, install the system libraries used by OpenCV and DOCX conversion:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libreoffice
```

Create an environment and install the service:

```bash
cd /srv/iarag/services/ocr
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[core]'
uvicorn main_API:app --host 0.0.0.0 --port 8084 --workers 1
```

MinerU selects CUDA when available and otherwise falls back to MPS, NPU, or CPU. CPU execution is possible but substantially slower for non-trivial documents.

## Model setup

The service bundles MinerU `2.6.2`, but not its model weights. By default, required pipeline model files are downloaded lazily from Hugging Face during the first OCR request. Consequently, the first request can take much longer than subsequent requests.

To pre-download the pipeline models into the current user's cache and generate `~/mineru.json`:

```bash
mineru-models-download --source huggingface --model_type pipeline
```

ModelScope is also supported:

```bash
mineru-models-download --source modelscope --model_type pipeline
```

For an offline deployment, set `MINERU_MODEL_SOURCE=local` and provide a MinerU configuration file whose `models-dir.pipeline` value points to the downloaded pipeline repository:

```json
{
  "models-dir": {
    "pipeline": "/absolute/path/to/PDF-Extract-Kit-1.0"
  }
}
```

The default configuration path is `~/mineru.json`. Set `MINERU_TOOLS_CONFIG_JSON` to an absolute path to use a different file. If models are downloaded inside a container, mount the model/cache and configuration paths into the container to preserve them when the container is recreated.

## Configuration

### Service settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_IDLE_UNLOAD_SECONDS` | `3600` | Idle time before cached MinerU models are unloaded. The next request reloads them. |
| `OCR_QUEUE_MAXSIZE` | `2` | Maximum number of waiting jobs. OCR execution remains single-file-at-a-time. |
| `OCR_QUEUE_RETRY_AFTER_SECONDS` | `30` | Value of the `Retry-After` header returned when the queue is full. |

These values are read when `main_API.py` is imported and must be valid integers.

### Relevant MinerU settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINERU_DEVICE_MODE` | auto-detect | Force a device such as `cuda`, `cuda:0`, or `cpu`. |
| `MINERU_MODEL_SOURCE` | `huggingface` | Model source: `huggingface`, `modelscope`, or `local`. |
| `MINERU_TOOLS_CONFIG_JSON` | `mineru.json` | MinerU configuration filename under the user's home directory, or an absolute path. |
| `MINERU_FORMULA_ENABLE` | `true` | Enable or disable formula recognition. |
| `MINERU_TABLE_ENABLE` | `false` | Enable or disable table recognition. The service disables it by default. |
| `MINERU_MIN_BATCH_INFERENCE_SIZE` | `384` | Pipeline batch threshold; increasing it may improve throughput while consuming more memory. |
| `HF_HOME` | Hugging Face default | Override the Hugging Face cache directory. Useful when mounting a persistent model cache. |

The Compose file exposes only the three `OCR_*` settings. Add MinerU variables and any required volume mounts under the `ocr` service when overriding model behavior.

## Processing behavior

1. The upload is copied into a unique temporary workspace.
2. DOCX input is converted to PDF with `soffice --headless`.
3. Text from the first PDF page is sampled for language detection.
4. MinerU runs in `pipeline` backend and `auto` parse mode, choosing direct extraction or OCR based on the document.
5. Markdown is generated with formula recognition enabled and table recognition disabled by default.
6. Image links, recognized contents/index blocks, and content from a top-level `References`, `Referencias`, or `Enlaces externos` heading onward are removed.
7. The response is returned and the temporary workspace is deleted.

Only one OCR job runs at a time. A disconnected client cancels its result; temporary data is still cleaned up by the worker. Generated Markdown and images are transient—the public API returns Markdown directly and removes its workspace after processing.

## Logs

The API emits JSON log events to standard output:

- `ocr_api_started` and `ocr_api_stopped` for lifecycle changes.
- `ocr_start` with queue wait time and request identifiers.
- `ocr_done` with queue wait and processing durations.
- `ocr_error` when document processing fails.
- `ocr_models_unloaded` when idle model cleanup releases cached models.

Follow container logs with:

```bash
docker compose logs -f ocr
```

## Project layout

```text
.
├── main_API.py          # FastAPI routes, queue worker, lifecycle, and logging
├── main_ocr.py          # PDF/DOCX processing and MinerU integration
├── utils.py             # Language detection and Markdown cleanup helpers
├── mineru/              # Bundled MinerU 2.6.2 source and resources
├── pyproject.toml       # Package metadata, dependencies, and CLI entry points
├── Dockerfile           # Python 3.12 runtime image
└── docker-compose.yml   # Single-GPU deployment on the external n8n network
```

## License

The package metadata declares the project under the AGPL-3.0 license. See the upstream MinerU project for its notices and third-party dependency terms.
