import json
import os
import time
from typing import Any, Dict, Iterable, List

from google import genai

from .config import VLM_MODEL, REQUEST_TIMEOUT, MAX_RETRIES, GEMINI_API_KEY


def _build_client() -> Any:
    api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not set')
    return genai.Client(api_key=api_key)


def _retryable(error: Exception) -> bool:
    msg = str(error).lower()
    return any(token in msg for token in ['rate limit', 'quota', '429', 'timeout', 'temporar', 'network'])


def gemini_extract_page(page_image_bytes: bytes, requested_fields: Iterable[str], page_number: int = 1) -> Dict[str, Any]:
    client = _build_client()
    field_list = list(requested_fields)
    prompt = {
        'text': (
            "Extract the requested shipping fields from this document page. "
            "Return only JSON with the exact keys requested. "
            "Each field must be a dict with keys: value, confidence, evidence. "
            "Do not invent missing values. Use null for missing values. "
            f"Requested fields: {field_list}."
        )
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=VLM_MODEL,
                contents=[prompt, {'inline_data': {'mime_type': 'image/png', 'data': page_image_bytes}}],
                config={'temperature': 0, 'response_mime_type': 'application/json', 'timeout': REQUEST_TIMEOUT},
            )
            payload = response.text if hasattr(response, 'text') else str(response)
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError('Gemini response was not a dict')
            return {'status': 'ok', 'data': parsed}
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if not _retryable(exc) or attempt >= MAX_RETRIES:
                return {'status': 'error', 'error': str(exc), 'attempts': attempt, 'page': page_number}
            time.sleep(2 ** attempt)
    if last_error:
        return {'status': 'error', 'error': str(last_error), 'attempts': MAX_RETRIES, 'page': page_number}
    return {'status': 'error', 'error': 'unknown gemini failure', 'attempts': 0, 'page': page_number}
