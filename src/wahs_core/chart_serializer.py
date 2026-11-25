"""Render condensed chart records into numbered `[N]` evidence lines and mappings."""

from __future__ import annotations

import re
from typing import Any, Optional

# Trim integer-valued trailing ".0" ("988.0" -> "988") without corrupting codes ("E11.0", "1.0.0").
_TRAILING_ZERO = re.compile(r"(?<![\w.])(\d+)\.0(?![\w.])")

_OBS_GROUP_UUID = "obs_group_uuid"
_OBS_GROUP_NAME = "obs_group_concept_name"
_CLINICAL_EVENT = "clinical_event"


def render_chart(records: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Render structured records to ``(chart_text, mappings)``.

    Each record renders as ``[N] (date) text (part of: <panel>) [dateKind]`` where the
    trailing ``[dateKind]`` marker appears only for non-clinical dates. Returns dense
    1-based indices paired with traceability mappings.
    """
    lines: list[str] = []
    mappings: list[dict[str, Any]] = []
    index = 0
    for rec in records:
        if not rec or not rec.get("resourceType") or not rec.get("resourceUuid"):
            continue
        index += 1
        date, is_clinical = _display_date(rec)
        body = _trim_zero(rec.get("text") or "")
        metadata = rec.get("metadata") or {}
        group = _group_label(metadata)
        date_prefix = f"({date}) " if date else ""
        marker = "" if is_clinical or not date else f" [{rec.get('dateKind') or 'unknown'}]"
        grounding_text = f"{date_prefix}{body}{group}"
        lines.append(f"[{index}] {grounding_text}{marker}")
        mapping_entry: dict[str, Any] = {
            "index": index,
            "resourceType": rec.get("resourceType"),
            "resourceUuid": rec.get("resourceUuid"),
            "date": date,
            "text": grounding_text,
        }
        if metadata.get("memberUuids"):
            mapping_entry["memberUuids"] = list(metadata["memberUuids"])
        mappings.append(mapping_entry)
    return ("\n".join(lines) + "\n" if lines else ""), mappings


def _display_date(rec: dict[str, Any]) -> tuple[Optional[str], bool]:
    clinical_date = rec.get("clinicalDate")
    if clinical_date:
        return clinical_date, True
    date_kind = rec.get("dateKind")
    if date_kind is None or date_kind == _CLINICAL_EVENT:
        return rec.get("date"), True
    return rec.get("date"), False


def _trim_zero(text: str) -> str:
    return _TRAILING_ZERO.sub(r"\1", text)


def _group_label(metadata: dict[str, Any]) -> str:
    if not metadata.get(_OBS_GROUP_UUID):
        return ""
    name = str(metadata.get(_OBS_GROUP_NAME) or "").strip()
    return f" (part of: {name})" if name else ""
