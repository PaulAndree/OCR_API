import copy, json, os, re, unicodedata
from pathlib import Path
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from PyPDF2 import PdfReader


OUTPUT_PATH = "/tmp/output_ocr"


def detect_language(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages[:1]:
        text += page.extract_text() or ""

    sample = " ".join(text.split()[:50])
    try:
        lang = detect(sample) if len(sample) > 0 else ""
    except LangDetectException:
        lang = ""

    if lang.startswith("es"):
        return "latin"
    else:
        return "en"
   
def title_pages_retrieval(file_name):
    aux_name = Path(file_name).stem
    aux = f"{aux_name}/auto/{aux_name}.json"
    path_json = os.path.join(OUTPUT_PATH, aux)

    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    combined_titles = []
    page_list = []
    i = 0
    
    while i < len(data):
        item = data[i]
        if item.get("text_level") == 1: # Only consider items with text_level == 1 (titles)
            combined_text = item["text"]
            page = item["page_idx"]
            j = i + 1 # Check for consecutive titles
            while j < len(data) and data[j].get("text_level") == 1:
                combined_text += " " + data[j]["text"]
                j += 1
            combined_titles.append(combined_text)
            page_list.append(page)
            i = j  # jump ahead to next non-consecutive item
        else:
            i += 1
            
    combined_titles = [" ".join(t.strip().lower().split()) for t in combined_titles]
    return combined_titles, page_list

def split_md_by_titles(markdown_text: str, file_name):
    sections = []
    current_titles = []
    current_content = []
    buffer_intro = []
    
    lines = markdown_text.splitlines()
    aux_title = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if aux_title != "" and current_content: # if there were previously titles and content save it first
                #block = "\n".join(current_titles) + "\n" + "\n".join(current_content).strip()
                current_titles.append(aux_title)
                block = aux_title + "\n" + "\n".join(current_content).strip()
                sections.append(block.strip())
                aux_title = ""
                current_content = []
            # else append the title found
            aux_title = aux_title + " " +  stripped.lstrip("#").strip()
            
        elif stripped == "": # empty lines -> skipp if there is previously titles but not content
            if aux_title != "" and not current_content:
                continue
            else:
                if aux_title != "":
                    current_content.append(line)
                else:
                    buffer_intro.append(line)
        else: # normal content
            if aux_title != "":
                current_content.append(line)
            else:
                buffer_intro.append(line)
                
    if aux_title != "" and current_content:  # append last section
        #block = "\n".join(current_titles) + "\n" + "\n".join(current_content).strip()
        current_titles.append(aux_title)
        block = aux_title + "\n" + "\n".join(current_content).strip()
        sections.append(block.strip())
        
    if buffer_intro:     # append the intro or first paragraph even without a title
        intro = "\n".join([ln for ln in buffer_intro if ln.strip() != ""]).strip()
        if intro:
            sections.insert(0, intro)
            current_titles.insert(0, " ")
            
    cleaned_sections = []     # check for empty lines and remove them
    for sec in sections:
        lines = [ln.rstrip() for ln in sec.splitlines() if ln.strip() != ""]
        cleaned_sections.append("\n".join(lines))

    #call to retrieve the page number detected in the OCR stage
    title_list, page_list = title_pages_retrieval(file_name)
    
    page_number_list = []
    
    for header in current_titles:
        header = " ".join(header.strip().lower().split())
        if not header:
            page_number_list.append(0)
        elif header in title_list:
            position = title_list.index(header)
            page_number_list.append(page_list[position])
        else:
            page_number_list.append(0)
    return current_titles, cleaned_sections, page_number_list



CONTENTS_TITLES = {
    "contents", "table of contents", "index", "indice",
    "tabla de contenidos", "sumario",
}


def _normalized_entry(line):
    text = re.sub(r'^\s*#{1,6}\s*', '', line)
    text = re.sub(r'\s*#+\s*$', '', text).strip()
    text = re.sub(
        r'(?:\.{2,}|\s+)\s*(?:\d+|[ivxlcdm]+)\s*$',
        '', text, flags=re.IGNORECASE
    )
    text = unicodedata.normalize('NFKD', text.casefold())
    text = ''.join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.strip(" .:-").split())


def _heading_entry(line):
    if not re.match(r'^\s*#{1,6}\s+\S', line):
        return None
    return _normalized_entry(line)


def _looks_like_contents_entry(line):
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        re.search(
            r'(?:\.{2,}|\s)\s*(?:\d+|[ivxlcdm]+)\s*$',
            stripped, re.IGNORECASE
        )
        or re.match(r'^\s*#{1,6}\s+', stripped)
    )


def _remove_contents(markdown_text):
    lines = markdown_text.splitlines()
    index = 0
    while index < len(lines):
        title = _heading_entry(lines[index])
        if title not in CONTENTS_TITLES:
            index += 1
            continue

        entries = set()
        entry_count = 0
        boundary = None
        for position in range(index + 1, len(lines)):
            normalized = _normalized_entry(lines[position])
            heading = _heading_entry(lines[position])
            if heading and heading in entries and entry_count >= 2:
                boundary = position
                break
            if normalized and len(normalized) < 180:
                entries.add(normalized)
            if _looks_like_contents_entry(lines[position]):
                entry_count += 1

        if boundary is not None:
            del lines[index:boundary]
            continue

        if title in {"index", "indice"}:
            tail = [line for line in lines[index + 1:] if line.strip()]
            index_entries = sum(_looks_like_contents_entry(line) for line in tail)
            has_later_heading = any(_heading_entry(line) for line in tail)
            if index_entries >= 3 and not has_later_heading:
                del lines[index:]
                break

        index += 1

    return "\n".join(lines)


def remove_images_n_references(markdown_text):
    # Remove all ![](...) patterns
    cleaned = re.sub(r'!\[\]\([^)]*\)', '', markdown_text)

    cleaned = _remove_contents(cleaned)

    # Pattern for references / external links section
    section_pattern = re.compile(r'^#\s*(References|Referencias|Enlaces externos)', re.IGNORECASE)

    filtered_lines = []
    for line in cleaned.splitlines():
        # Stop copying lines when we reach the References section
        if section_pattern.match(line.strip()):
            break
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()

