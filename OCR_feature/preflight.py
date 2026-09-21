import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from .config import GOOGLE_APPLICATION_CREDENTIALS, VLM_MODEL
from .pipeline import extract


def check_google_setup() -> None:
    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise RuntimeError(
            "Missing GOOGLE_APPLICATION_CREDENTIALS. Export the path to your Google Cloud service-account JSON key before running OCR."
        )
    if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {GOOGLE_APPLICATION_CREDENTIALS}"
        )
    try:
        from google.cloud import vision

        client = vision.ImageAnnotatorClient()
        img = Image.new('RGB', (300, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((12, 40), 'Port of Loading: PORT KLANG', fill=(0, 0, 0))
        buf = __import__('io').BytesIO()
        img.save(buf, format='PNG')
        req = vision.Image(content=buf.getvalue())
        response = client.document_text_detection(image=req)
        if response.error.message:
            raise RuntimeError(response.error.message)
    except Exception as exc:
        raise RuntimeError(f"Google Vision API preflight failed: {exc}") from exc

    if not os.getenv('GEMINI_API_KEY'):
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it in the environment before using the VLM fallback."
        )


def main() -> int:
    try:
        check_google_setup()
        print('Google Cloud preflight OK')
        print(f'Vision credentials: {GOOGLE_APPLICATION_CREDENTIALS}')
        print(f'Version: {VLM_MODEL}')
        sample = Path(__file__).resolve().parents[2] / 'data_v2' / 'attachments' / 'email_001_SI.txt'
        if sample.exists():
            print('Sample extraction:')
            result = extract(str(sample))
            print(result['status'])
            print(result['fields'])
        return 0
    except Exception as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        print('Google Cloud setup required:', file=sys.stderr)
        print('1. Enable the Cloud Vision API for the project.', file=sys.stderr)
        print('2. Create a Google Cloud service account and JSON key.', file=sys.stderr)
        print('3. Set GOOGLE_APPLICATION_CREDENTIALS to the JSON file path.', file=sys.stderr)
        print('4. Set GEMINI_API_KEY for Gemini fallback.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
