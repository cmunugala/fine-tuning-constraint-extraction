"""Structural, semantic, and leakage validation."""

from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import re
from typing import Any, Iterable

from models import SchedulingOutput

from .catalog import TEMPORAL_TYPES, WEEKLY_TYPES


def canonical_label(label: dict[str, Any]) -> str:
    return json.dumps(label, sort_keys=True, separators=(",", ":"))


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def request_skeleton(record: dict[str, Any]) -> str:
    """Normalize generated slot values while preserving linguistic structure."""

    text = normalized_text(record["text"])
    employees = {
        constraint.get("employee")
        for constraint in record["label"]["constraints"]
        if constraint.get("employee")
    }
    for employee in sorted(employees, key=len, reverse=True):
        text = re.sub(re.escape(employee.lower()), "<employee>", text)
    text = text.replace("everyone on the team", "<employee>")
    text = text.replace("each team member", "<employee>")
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "<date>", text)
    months = (
        "january|february|march|april|may|june|july|august|september|"
        "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )
    text = re.sub(rf"\b(?:{months})\s+\d{{1,2}},\s+\d{{4}}\b", "<date>", text)
    text = re.sub(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mons?|tues?|weds?|thus?|fris?|sats?|suns?)\b",
        "<day>",
        text,
    )
    text = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", "<time>", text)
    text = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", "<time>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", text)
    return re.sub(r"\s+", " ", text)


def validate_output(output: SchedulingOutput) -> None:
    if not output.constraints:
        raise ValueError("A label must contain at least one constraint")

    weekly: dict[str | None, dict[str, float]] = {}
    for constraint in output.constraints:
        if constraint.type == "available" and constraint.priority != "hard":
            raise ValueError("available constraints must be hard")
        if constraint.type == "preferred_hours" and constraint.priority != "soft":
            raise ValueError("preferred_hours constraints must be soft")
        if constraint.type in TEMPORAL_TYPES:
            if constraint.time_scope is None:
                raise ValueError(f"{constraint.type} requires time_scope")
            if constraint.value is not None:
                raise ValueError(f"{constraint.type} must not have value")
        else:
            if constraint.value is None or constraint.value <= 0:
                raise ValueError(f"{constraint.type} requires a positive value")
            if constraint.time_scope is not None:
                raise ValueError(f"{constraint.type} must not have time_scope")

        scope = constraint.time_scope
        if scope is not None:
            if scope.start_time and scope.end_time and scope.start_time >= scope.end_time:
                raise ValueError("start_time must precede end_time")
            if scope.start_date and scope.end_date and scope.start_date > scope.end_date:
                raise ValueError("start_date must not follow end_date")
            if scope.days_of_week and len(scope.days_of_week) != len(set(scope.days_of_week)):
                raise ValueError("days_of_week must not contain duplicates")
            if scope.recurring and not scope.days_of_week:
                raise ValueError("recurring scopes require days_of_week")
            if not scope.recurring and not scope.start_date:
                raise ValueError("one-off scopes require a date")

        if constraint.type in WEEKLY_TYPES:
            weekly.setdefault(constraint.employee, {})[constraint.type] = constraint.value

    for limits in weekly.values():
        minimum = limits.get("min_weekly_hours")
        maximum = limits.get("max_weekly_hours")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("min_weekly_hours exceeds max_weekly_hours")


def validate_record(record: dict[str, Any]) -> SchedulingOutput:
    expected = {"text", "reference_date", "label"}
    if set(record) != expected:
        raise ValueError(f"Record keys must be exactly {sorted(expected)}")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("text must be a non-empty string")
    date.fromisoformat(record["reference_date"])
    output = SchedulingOutput.model_validate(record["label"])
    validate_output(output)
    return output


def validate_records(records: Iterable[dict[str, Any]]) -> list[SchedulingOutput]:
    return [validate_record(record) for record in records]


def audit_splits(
    records_by_split: dict[str, list[dict[str, Any]]],
    metadata_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    text_owners: dict[str, str] = {}
    label_owners: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    skeleton_counts: Counter[str] = Counter()

    for split, records in records_by_split.items():
        outputs = validate_records(records)
        for record, output in zip(records, outputs, strict=True):
            text_fp = fingerprint(normalized_text(record["text"]))
            label_fp = fingerprint(canonical_label(record["label"]))
            if text_fp in text_owners:
                raise ValueError(f"Duplicate request across {text_owners[text_fp]} and {split}")
            if label_fp in label_owners:
                raise ValueError(f"Duplicate label across {label_owners[label_fp]} and {split}")
            text_owners[text_fp] = split
            label_owners[label_fp] = split
            type_counts.update(item.type for item in output.constraints)
            priority_counts.update(item.priority for item in output.constraints)
            skeleton_counts[request_skeleton(record)] += 1

    style_counts: Counter[str] = Counter()
    if metadata_by_split is not None:
        for split, metadata in metadata_by_split.items():
            if len(metadata) != len(records_by_split[split]):
                raise ValueError(f"Metadata length does not match records for {split}")
            for item in metadata:
                style_counts[item["style"]] += 1

    result = {
        "counts": {split: len(records) for split, records in records_by_split.items()},
        "constraint_type_counts": dict(sorted(type_counts.items())),
        "constraint_priority_counts": dict(sorted(priority_counts.items())),
        "unique_texts": len(text_owners),
        "unique_labels": len(label_owners),
        "unique_text_skeletons": len(skeleton_counts),
        "max_text_skeleton_count": max(skeleton_counts.values(), default=0),
        "max_text_skeleton_share": (
            max(skeleton_counts.values(), default=0) / len(text_owners)
            if text_owners
            else 0.0
        ),
    }
    if metadata_by_split is not None:
        result["render_style_counts"] = dict(sorted(style_counts.items()))
    return result
