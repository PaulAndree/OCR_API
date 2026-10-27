import copy, gc, json, os
from pathlib import Path
import subprocess
from loguru import logger
from getpass import getpass
from utils import detect_language, title_pages_retrieval, split_md_by_titles, remove_images_n_references
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


OUTPUT_PATH = "/tmp/output_ocr"
#OUTPUT_PATH =  config.OUTPUT_DIR


def unload_models() -> bool:
    """Drop MinerU's cached models so their VRAM goes back to the GPU.

    MinerU keeps every model it loads in class-level singletons for the life of
    the process (about 2.4 GB after one PDF). Clearing those dicts removes the
    last references; the next request just loads them again. The CUDA context
    itself (a few hundred MB) stays until the process exits.
    Returns True if anything was loaded.
    """
    from mineru.backend.pipeline.pipeline_analyze import ModelSingleton as PipelineModels
    from mineru.backend.pipeline.model_init import AtomModelSingleton
    from mineru.utils.block_sort import ModelSingleton as ReadingOrderModels
    from mineru.utils.model_utils import clean_memory
    from mineru.utils.config_reader import get_device

    caches = [PipelineModels._models, AtomModelSingleton._models, ReadingOrderModels._models]
    if not any(caches):
        return False
    for cache in caches:
        cache.clear()
    gc.collect()  # free the tensors first, so empty_cache can hand their blocks back
    clean_memory(get_device())
    return True

def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_name: str,  # PDF file names to be parsed
    pdf_bytes: bytes,  # PDF bytes to be parsed
    p_lang: str,  # languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=False,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=False,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=False,  # Whether to dump middle JSON files
    f_dump_model_output=False,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=False,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    if backend == "pipeline":
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [new_pdf_bytes],  
            [p_lang],  # Wrap in list
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable
        )
        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            #pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)
            pdf_info = middle_json["pdf_info"]
            #pdf_bytes = pdf_bytes_list[idx]
            
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

                
            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                # borrar imagenes [url] y referencias
                md_content_str = remove_images_n_references(md_content_str)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )
            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )
            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )
            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )
            logger.info(f"local output dir is {local_md_dir}")

            return md_content_str
    else:
        print("Backend not found")

        
def parse_doc(
        path: Path,
        output_dir,
        lang="en",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    try:
        file_name = str(Path(path).stem)
        pdf_bytes = read_fn(path)

        do_parse_result = do_parse(
            output_dir=output_dir,
            pdf_file_name=file_name,
            pdf_bytes=pdf_bytes,
            p_lang=lang,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
        return do_parse_result
    except Exception as e:
        logger.exception(e)
        raise



class ocr_model:
    def __init__(self, output_path=OUTPUT_PATH):
        self.ocr_lan = ""
        self.base_output_path = output_path
        self.output_path = ""
        self.content_parsed = ""
          
    def get_pdf_text(self, doc_path, file_name):
        lang = detect_language(doc_path)
        self.ocr_lan = lang
        self.output_path = os.path.join(self.base_output_path, file_name, "auto", f"{file_name}.md")
        
        self.content_parsed = parse_doc(doc_path, self.base_output_path, lang=lang, backend="pipeline")


    def get_doc_text(self, doc_path, file_name):
        os.makedirs(self.base_output_path, exist_ok=True)
        # Convert DOCX to PDF using LibreOffice
        pdf_filename = os.path.splitext(os.path.basename(doc_path))[0] + ".pdf"
        output_pdf_path = os.path.join(self.base_output_path, pdf_filename)
        subprocess.run([
            "soffice", "--headless", "--convert-to", "pdf",
            "--outdir", self.base_output_path, doc_path
        ], check=True)
        lang = detect_language(output_pdf_path)
        self.ocr_lan = lang
        self.output_path = os.path.join(self.base_output_path, file_name, "auto", f"{file_name}.md")
        self.content_parsed = parse_doc(output_pdf_path, self.base_output_path, lang=lang, backend="pipeline")
     
    def as_dict(self):
        return {
            "ocr_language":self.ocr_lan,
            "content_parsed":self.content_parsed
        }
