import os
from typing import Any, Dict, List, Tuple

from google.cloud import vision

from .config import OCR_DPI, GOOGLE_APPLICATION_CREDENTIALS


def _resolve_google_credentials() -> None:
    if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS


def _word_to_dict(word: Any, page_number: int) -> Dict[str, Any]:
    vertices = word.bounding_box.vertices if getattr(word, 'bounding_box', None) else []
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return {
        'text': word.text,
        'confidence': float(getattr(word, 'confidence', 0.0) or 0.0),
        'page': page_number,
        'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)] if xs and ys else [0, 0, 0, 0],
    }


def ocr_page_image(image_bytes: bytes, page_number: int = 1) -> Dict[str, Any]:
    _resolve_google_credentials()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    pages: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for page in response.full_text_annotation.pages:
        page_lines: List[str] = []
        page_words: List[Dict[str, Any]] = []
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = ''.join(symbol.text for symbol in word.symbols)
                    if not text:
                        continue
                    word_dict = _word_to_dict(word, page_number)
                    word_dict['text'] = text
                    page_words.append(word_dict)
                    words.append(word_dict)
                if page_words:
                    sorted_words = sorted(page_words, key=lambda w: (w['bbox'][1], w['bbox'][0]))
                    current_line: List[Dict[str, Any]] = []
                    current_y: float | None = None
                    for item in sorted_words:
                        y = item['bbox'][1]
                        if current_y is None or abs(y - current_y) < 10:
                            current_line.append(item)
                            current_y = y
                        else:
                            line = ' '.join(w['text'] for w in sorted(current_line, key=lambda i: i['bbox'][0]))
                            page_lines.append(line)
                            current_line = [item]
                            current_y = y
                    if current_line:
                        page_lines.append(' '.join(w['text'] for w in sorted(current_line, key=lambda i: i['bbox'][0])))
                    page_words = []
        pages.append({'page': page_number, 'text': '\n'.join(page_lines), 'words': words, 'lines': page_lines})
    result_text = '\n'.join(page['text'] for page in pages)
    return {'text': result_text, 'pages': pages, 'words': words}


def ocr_file(path: str, page_number: int = 1) -> Dict[str, Any]:
    with open(path, 'rb') as fh:
        image_bytes = fh.read()
    return ocr_page_image(image_bytes, page_number=page_number)
