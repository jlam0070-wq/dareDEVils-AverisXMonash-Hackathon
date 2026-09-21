import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import (
    CONFIDENCE_THRESHOLD,
    FIELD_ORDER,
    MAX_RETRIES,
    OCR_DPI,
    REQUEST_TIMEOUT,
    VLM_MODEL,
)
from .fields import extract_fields_from_text
from .native_readers import detect_file_type, read_native_document, render_pdf_page_to_image
from .ocr import ocr_page_image
from .vlm import gemini_extract_page


@dataclass
class FieldResult:
    value: Any = None
    confidence: float = 0.0
    method: str = 'native_text'
    evidence: Dict[str, Any] = field(default_factory=dict)
    needs_review: bool = False
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'confidence': self.confidence,
            'method': self.method,
            'evidence': self.evidence,
            'needs_review': self.needs_review,
            'reason': self.reason,
        }


@dataclass
class DocumentResult:
    document_id: str
    path: str
    file_type: str
    status: str = 'OK'
    pages: int = 0
    warnings: List[str] = field(default_factory=list)
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_dict(self):
        return {
            'document_id': self.document_id,
            'path': self.path,
            'file_type': self.file_type,
            'status': self.status,
            'pages': self.pages,
            'warnings': self.warnings,
            'fields': self.fields,
        }


def _safe_read_document(path: str) -> tuple[str, Dict[str, Any]]:
    file_type = detect_file_type(path)
    if file_type == 'unknown':
        return 'UNREADABLE', {'status': 'UNREADABLE', 'reason': f'Unsupported file type for {Path(path).suffix}'}
    try:
        native = read_native_document(path)
        if native.get('error'):
            return 'UNREADABLE', {'status': 'UNREADABLE', 'reason': native['error']}
        return file_type, {'status': 'OK', 'pages': native.get('pages', [])}
    except Exception as exc:
        return 'UNREADABLE', {'status': 'UNREADABLE', 'reason': str(exc)}


def _merge_field(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in new.items():
        if value is not None and (key not in merged or merged[key] is None):
            merged[key] = value
    return merged


def _build_needs_review(reason: str, evidence: Dict[str, Any], value=None, confidence=0.0, method='native_text') -> Dict[str, Any]:
    return {'value': value, 'confidence': confidence, 'method': method, 'evidence': evidence, 'needs_review': True, 'reason': reason}


def _field_from_ocr_value(value: Any, page: int, source: str) -> Dict[str, Any]:
    return {'value': value, 'confidence': 0.9, 'method': 'ocr', 'evidence': {'page': page, 'source_text': source}, 'needs_review': False, 'reason': None}


def _parse_native_fields(text: str, fields: Iterable[str] | None = None) -> Dict[str, Any]:
    data = extract_fields_from_text(text, fields)
    result = {}
    for field, item in data.items():
        result[field] = item
    return result


def _extract_with_vlm(page_image_bytes: bytes, requested_fields: Iterable[str], page_number: int = 1) -> Optional[Dict[str, Any]]:
    result = gemini_extract_page(page_image_bytes, requested_fields, page_number)
    if result.get('status') != 'ok':
        return None
    return result['data']


def _candidate_validated(field: str, candidate: Any, page_text: str) -> bool:
    if candidate is None:
        return False
    text = str(candidate).lower()
    if not text:
        return False
    if text in {'null', 'none'}:
        return False
    return text in page_text.lower() or page_text.lower().find(text) >= 0


def extract(path: str, fields: Iterable[str] | None = None, email_body: str | None = None) -> Dict[str, Any]:
    path_obj = Path(path)
    doc_id = path_obj.stem
    file_type = detect_file_type(path_obj)
    allowed_fields = list(fields) if fields else list(FIELD_ORDER)
    status = 'OK'
    warnings: List[str] = []
    all_pages: List[Dict[str, Any]] = []
    native_fields: Dict[str, Any] = {}

    if file_type == 'unknown':
        return {
            'document_id': doc_id,
            'path': str(path_obj),
            'file_type': file_type,
            'status': 'UNREADABLE',
            'pages': 0,
            'warnings': ['Unsupported file type'],
            'fields': {field: {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {}, 'needs_review': True, 'reason': 'unsupported file type'} for field in allowed_fields},
        }

    try:
        native_doc = read_native_document(path_obj)
        if native_doc.get('error'):
            raise ValueError(native_doc['error'])
        all_pages = native_doc.get('pages', [])
    except Exception as exc:
        return {
            'document_id': doc_id,
            'path': str(path_obj),
            'file_type': file_type,
            'status': 'UNREADABLE',
            'pages': 0,
            'warnings': [str(exc)],
            'fields': {field: {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {}, 'needs_review': True, 'reason': str(exc)} for field in allowed_fields},
        }

    for page in all_pages:
        page_text = page.get('text', '') or ''
        if page_text:
            native_fields.update(_parse_native_fields(page_text, allowed_fields))

    if not any((item.get('value') is not None and item.get('value') != '') for item in native_fields.values()):
        if file_type in {'pdf', 'image'} or all_pages and any(page.get('is_scanned') for page in all_pages):
            ocr_results: List[Dict[str, Any]] = []
            for page_num, page in enumerate(all_pages, start=1):
                if page.get('is_scanned') or page.get('text', '').strip() == '':
                    try:
                        payload = ocr_page_image(render_pdf_page_to_image(path_obj, page_num, dpi=OCR_DPI), page_num)
                    except Exception as exc:
                        warnings.append(f'OCR failed on page {page_num}: {exc}')
                        continue
                    ocr_results.append(payload)
            combined_text = '\n'.join(result.get('text', '') for result in ocr_results)
            if combined_text:
                native_fields.update(_parse_native_fields(combined_text, allowed_fields))

    for field in allowed_fields:
        item = native_fields.get(field)
        if item is None or item.get('value') is None or item.get('value') == '':
            if file_type in {'pdf', 'image'} and all_pages:
                for page_num, page in enumerate(all_pages, start=1):
                    page_text = page.get('text', '') or ''
                    if not page_text and page_num:
                        try:
                            image_bytes = render_pdf_page_to_image(path_obj, page_num, dpi=OCR_DPI)
                        except Exception as exc:
                            warnings.append(f'Cannot render page {page_num}: {exc}')
                            continue
                        result = _extract_with_vlm(image_bytes, [field], page_num)
                        if result and field in result:
                            candidate = result[field].get('value') if isinstance(result[field], dict) else result[field]
                            if candidate is not None and _candidate_validated(field, candidate, page_text):
                                native_fields[field] = {'value': candidate, 'confidence': float(result[field].get('confidence', 0.0) if isinstance(result[field], dict) else 0.0), 'method': 'vlm', 'evidence': {'page': page_num, 'source_text': str(candidate)}, 'needs_review': False, 'reason': None}
                                break
                            native_fields[field] = {'value': None, 'confidence': 0.0, 'method': 'vlm', 'evidence': {'page': page_num, 'source_text': page_text or ''}, 'needs_review': True, 'reason': 'VLM value not verified in document text'}
                            break
                        native_fields[field] = {'value': None, 'confidence': 0.0, 'method': 'vlm', 'evidence': {'page': page_num, 'source_text': ''}, 'needs_review': True, 'reason': 'missing field after VLM fallback'}
                        break
            else:
                native_fields[field] = {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {'source_text': ''}, 'needs_review': True, 'reason': 'missing field'}

    for field in allowed_fields:
        item = native_fields.get(field)
        if not item:
            native_fields[field] = {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {'source_text': ''}, 'needs_review': True, 'reason': 'missing field'}
            continue
        if field in {'container_count', 'gross_weight_kg'} and item.get('value') is not None:
            parsed = item['value']
            if isinstance(parsed, str):
                value = re.sub(r'[^0-9.\-]', '', parsed)
                if not value:
                    item['needs_review'] = True
                    item['reason'] = 'weight or container values cannot be parsed'
                    item['value'] = None
                    item['confidence'] = 0.0
                else:
                    item['value'] = float(value)
                    item['confidence'] = max(float(item.get('confidence', 0.0)), 0.0)
        if item.get('needs_review'):
            status = 'NEEDS_REVIEW'

    result = {
        'document_id': doc_id,
        'path': str(path_obj),
        'file_type': file_type,
        'status': status,
        'pages': len(all_pages),
        'warnings': warnings,
        'fields': {field: native_fields.get(field, {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {}, 'needs_review': True, 'reason': 'missing field'}) for field in allowed_fields},
    }
    if email_body:
        body = email_body or ''
        for field in allowed_fields:
            item = result['fields'].get(field)
            if item and item.get('value') is None and body:
                match = re.search(rf'{field.replace("_", " ")}:?\s*([^\n]+)', body, flags=re.I)
                if match:
                    item['value'] = match.group(1).strip()
                    item['method'] = 'email_body'
                    item['evidence'] = {'source_text': match.group(0), 'location': 'email_body'}
                    item['needs_review'] = False
    return result


def batch_extract(paths: Iterable[str], email_body: str | None = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in paths:
        results.append(extract(path, email_body=email_body))
    return results


def to_label_text(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    for field in FIELD_ORDER:
        item = (result.get('fields') or {}).get(field)
        if item and item.get('value') is not None:
            lines.append(f'{field}: {item["value"]}')
    return '\n'.join(lines)
