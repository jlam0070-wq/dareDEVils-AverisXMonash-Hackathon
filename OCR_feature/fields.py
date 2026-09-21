import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..comparison.processing_steps.normalization import normalize_label

from .config import FIELD_ORDER, FIELD_LABELS


FIELD_ALIAS_MAP: Dict[str, set[str]] = {
    'shipper': {'shipper', 'shipper exporter', 'principal or seller'},
    'consignee': {'consignee', 'to the order of', 'to the order of shipper'},
    'notify_party': {'notify party', 'notify', 'intermediate consignee'},
    'port_of_loading': {'port of loading', 'load port', 'pol', 'loading port'},
    'port_of_discharge': {'port of discharge', 'discharge port', 'pod'},
    'container_count': {'container count', 'total containers', 'no of containers', 'no containers', 'number of containers'},
    'gross_weight_kg': {'gross weight', 'gross wt', 'gross weight kg', 'gross weight kgs'},
}


def canonical_field_name(name: str) -> str | None:
    key = normalize_label(name or '')
    if not key:
        return None
    for field, aliases in FIELD_ALIAS_MAP.items():
        if key in {normalize_label(alias) for alias in aliases}:
            return field
    return None


def normalize_value(value: str | int | float | None) -> str:
    text = str(value or '').strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def coerce_numeric(field: str, raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None

    if field == 'container_count':
        match = re.search(r'(\d{1,4})\s*(?:x|X)\s*(\d{1,4})', text)
        if match:
            return int(match.group(1))
        nums = re.findall(r'\d+', text)
        return int(nums[0]) if nums else None

    numbers = [int(n.replace(',', '')) for n in re.findall(r'\d[\d,]*', text)]
    if not numbers:
        return None
    return max(numbers)


def _label_variants_found(text: str) -> List[str]:
    matches: List[str] = []
    for label in re.findall(r'(?i)(?:[A-Za-z][A-Za-z0-9/() .-]{2,60})\s*[:\|]', text):
        candidate = label.rstrip(':|').strip()
        if candidate and candidate.lower() not in {'bill of lading', 'shipping instruction', 'shippper'}:
            matches.append(candidate)
    return matches


def extract_fields_from_text(text: str | None, fields: Iterable[str] | None = None) -> Dict[str, Dict[str, Any]]:
    text = text or ''
    lines = [ln.rstrip() for ln in text.splitlines()]
    field_names = set(fields) if fields else set(FIELD_ORDER)
    found: Dict[str, Dict[str, Any]] = {}
    current_field: str | None = None

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        label_match = re.match(r'^([^:|]+?)\s*[:|]\s*(.*)$', raw)
        if label_match:
            label, value = label_match.group(1).strip(), label_match.group(2).strip()
            field = canonical_field_name(label) or (
                next((f for f in FIELD_ORDER if label.lower().startswith(f.replace('_', ' '))), None)
            )
            if field and field in field_names:
                current_field = field
                if value:
                    found[field] = {'value': value, 'confidence': 1.0, 'method': 'native_text', 'evidence': {'source_text': raw}, 'needs_review': False, 'reason': None}
                continue
            current_field = None
            continue

        if current_field and current_field in field_names:
            entry = found.get(current_field)
            if entry:
                entry['value'] = f"{entry['value']} {raw}".strip()
                entry['evidence']['source_text'] = f"{entry['evidence']['source_text']} | {raw}"
            else:
                found[current_field] = {'value': raw, 'confidence': 1.0, 'method': 'native_text', 'evidence': {'source_text': raw}, 'needs_review': False, 'reason': None}
            continue

        matched = None
        for field in FIELD_ORDER:
            if field not in field_names:
                continue
            aliases = {normalize_label(a) for a in FIELD_ALIAS_MAP[field]}
            if normalize_label(raw) in aliases or any(normalize_label(raw).startswith(a + ' ') for a in aliases):
                matched = field
                break
        if matched:
            current_field = matched
            continue

        current_field = None

    for field in list(found.keys()):
        item = found[field]
        item['value'] = str(item['value']).strip()
        if field in {'container_count', 'gross_weight_kg'}:
            numeric = coerce_numeric(field, item['value'])
            if numeric is None:
                item['needs_review'] = True
                item['reason'] = 'weight or container values cannot be parsed'
                item['value'] = None
            else:
                item['value'] = numeric
        if item.get('value') is None:
            item['confidence'] = 0.0

    return {
        field: found.get(
            field,
            {'value': None, 'confidence': 0.0, 'method': 'native_text', 'evidence': {'source_text': ''}, 'needs_review': True, 'reason': 'missing field'},
        )
        for field in FIELD_ORDER if field in field_names
    }
