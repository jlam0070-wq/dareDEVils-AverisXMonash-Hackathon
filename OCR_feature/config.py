import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
MIN_NATIVE_TEXT_CHARS_PER_PAGE = int(os.getenv("MIN_NATIVE_TEXT_CHARS_PER_PAGE", "120"))
VLM_MODEL = os.getenv("VLM_MODEL", "gemini-2.5-flash")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

FIELD_ORDER = [
    "shipper",
    "consignee",
    "notify_party",
    "port_of_loading",
    "port_of_discharge",
    "container_count",
    "gross_weight_kg",
]

FIELD_LABELS = {
    "shipper": ["shipper", "shipper/exporter", "exporter", "principal or seller"],
    "consignee": ["consignee", "to the order of", "to the order of shipper"],
    "notify_party": ["notify party", "notify", "intermediate consignee"],
    "port_of_loading": ["port of loading", "load port", "loading port", "pol"],
    "port_of_discharge": ["port of discharge", "discharge port", "pod"],
    "container_count": ["container count", "total containers", "no of containers", "no containers"],
    "gross_weight_kg": ["gross weight", "gross wt", "gross weight kg", "gross weight kgs"],
}
