"""Dataset orchestration, review gating, and atomic JSONL output."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .catalog import (
    CONSTRAINT_TYPES,
    HELD_OUT_COMBINATIONS,
    SEEN_COMBINATIONS,
    TYPE_WEIGHTS,
    combination_key,
)
from .generation import Scenario, make_scenario
from .rendering import assert_template_banks_disjoint, render_scenario
from .validation import (
    audit_splits,
    canonical_label,
    fingerprint,
    normalized_text,
    validate_record,
    validate_records,
)

DEFAULT_SEED = 20261102
DEFAULT_SIZES = {"train": 1000, "validation": 150, "test": 200}
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}


def _record(scenario: Scenario, split: str, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    text, template_ids = render_scenario(scenario, split, rng)
    label = scenario.output.model_dump(mode="json", exclude_none=True)
    record = {
        "text": text,
        "reference_date": scenario.reference_date.isoformat(),
        "label": label,
    }
    validate_record(record)
    metadata = {
        "template_ids": list(template_ids),
        "combination": list(combination_key([item.type for item in scenario.output.constraints])),
    }
    return record, metadata


def _weighted_single(rng: random.Random) -> tuple[str, ...]:
    return (rng.choices(CONSTRAINT_TYPES, weights=TYPE_WEIGHTS, k=1)[0],)


def _plans_for_split(split: str, size: int, rng: random.Random) -> list[tuple[str, ...]]:
    single_count = math.floor(size * 0.55 + 0.5)
    multi_count = size - single_count
    plans = [_weighted_single(rng) for _ in range(single_count)]

    held_out_count = min(multi_count, math.floor(size * 0.30 + 0.5)) if split == "test" else 0
    for index in range(held_out_count):
        plans.append(HELD_OUT_COMBINATIONS[index % len(HELD_OUT_COMBINATIONS)])
    for index in range(multi_count - held_out_count):
        # Cycling guarantees every known combination appears in sufficiently large splits.
        plans.append(SEEN_COMBINATIONS[index % len(SEEN_COMBINATIONS)])
    rng.shuffle(plans)
    return plans


def _generate_split(
    split: str,
    size: int,
    rng: random.Random,
    used_texts: set[str],
    used_labels: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    plans = _plans_for_split(split, size, rng)

    for types in plans:
        accepted = False
        for _ in range(250):
            team_wide = rng.random() < 0.10
            multiple_employees = len(types) > 1 and not team_wide and rng.random() < 0.10
            scenario = make_scenario(
                types,
                rng,
                team_wide=team_wide,
                multiple_employees=multiple_employees,
            )
            record, item_metadata = _record(scenario, split, rng)
            text_fp = fingerprint(normalized_text(record["text"]))
            label_fp = fingerprint(canonical_label(record["label"]))
            if text_fp in used_texts or label_fp in used_labels:
                continue
            used_texts.add(text_fp)
            used_labels.add(label_fp)
            records.append(record)
            metadata.append(item_metadata)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Could not generate a unique {split} record for {types}")
    return records, metadata


PREVIEW_PLANS: tuple[dict[str, Any], ...] = (
    {"types": ("unavailable",), "styles": ("recurring",), "priorities": ("hard",)},
    {
        "types": ("available",),
        "styles": ("one_off",),
        "priorities": ("hard",),
        "date_offsets": (1,),
    },
    {"types": ("preferred_hours",), "styles": ("recurring",), "priorities": ("soft",)},
    {"types": ("min_weekly_hours",), "priorities": ("hard",)},
    {"types": ("max_weekly_hours",), "priorities": ("soft",)},
    {"types": ("max_consecutive_days",), "priorities": ("hard",)},
    {"types": ("min_rest_hours",), "priorities": ("hard",)},
    {"types": ("unavailable",), "styles": ("bounded",), "priorities": ("soft",)},
    {"types": ("available",), "styles": ("bounded",), "priorities": ("hard",)},
    {"types": ("max_weekly_hours",), "team_wide": True, "priorities": ("hard",)},
    {
        "types": ("unavailable",),
        "styles": ("one_off",),
        "priorities": ("hard",),
        "date_offsets": (4,),
    },
    {"types": ("min_weekly_hours", "max_weekly_hours")},
    {"types": ("unavailable", "max_weekly_hours")},
    {"types": ("available", "preferred_hours"), "styles": ("recurring", "recurring")},
    {"types": ("max_consecutive_days", "min_rest_hours")},
    {"types": ("unavailable", "max_weekly_hours", "min_rest_hours")},
    {"types": ("preferred_hours", "min_weekly_hours"), "multiple_employees": True},
    {"types": ("unavailable", "min_rest_hours"), "priorities": ("soft", "hard")},
    {"types": ("available", "min_weekly_hours"), "team_wide": True},
    {"types": ("preferred_hours", "max_weekly_hours", "max_consecutive_days")},
)


def build_preview(seed: int = DEFAULT_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_template_banks_disjoint()
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    used_texts: set[str] = set()
    used_labels: set[str] = set()

    for index, plan in enumerate(PREVIEW_PLANS):
        # The first week of November gives preview rows predictable relative-date examples.
        reference_date = date(2026, 11, 2) if index in {1, 10} else None
        scenario = make_scenario(
            plan["types"],
            rng,
            reference_date=reference_date,
            team_wide=plan.get("team_wide", False),
            multiple_employees=plan.get("multiple_employees", False),
            scope_styles=plan.get("styles"),
            priorities=plan.get("priorities"),
            one_off_offsets=plan.get("date_offsets"),
        )
        record, item_metadata = _record(scenario, "preview", rng)
        text_fp = fingerprint(normalized_text(record["text"]))
        label_fp = fingerprint(canonical_label(record["label"]))
        if text_fp in used_texts or label_fp in used_labels:
            raise RuntimeError("Preview unexpectedly contains a duplicate")
        used_texts.add(text_fp)
        used_labels.add(label_fp)
        records.append(record)
        metadata.append(item_metadata)

    stats = audit_splits({"preview": records}, {"preview": metadata})
    return records, {
        "dataset": "preview",
        "seed": seed,
        "schema_model": "models.SchedulingOutput",
        "status": "awaiting_manual_approval",
        "unique_template_ids": len(
            {template_id for item in metadata for template_id in item["template_ids"]}
        ),
        **stats,
    }


def build_full_dataset(
    seed: int = DEFAULT_SEED,
    sizes: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    assert_template_banks_disjoint()
    sizes = dict(sizes or DEFAULT_SIZES)
    if set(sizes) != {"train", "validation", "test"}:
        raise ValueError("sizes must define train, validation, and test")
    rng = random.Random(seed)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    metadata_by_split: dict[str, list[dict[str, Any]]] = {}
    used_texts: set[str] = set()
    used_labels: set[str] = set()

    for split in ("train", "validation", "test"):
        records, metadata = _generate_split(
            split, sizes[split], rng, used_texts, used_labels
        )
        records_by_split[split] = records
        metadata_by_split[split] = metadata

    stats = audit_splits(records_by_split, metadata_by_split)
    held_out_test_count = sum(
        combination_key(item["combination"])
        in {combination_key(combo) for combo in HELD_OUT_COMBINATIONS}
        for item in metadata_by_split["test"]
    )
    manifest = {
        "dataset": "scheduling_constraints",
        "seed": seed,
        "schema_model": "models.SchedulingOutput",
        "row_fields": ["text", "reference_date", "label"],
        "null_fields_omitted": True,
        "split_policy": {
            "single_constraint_ratio": 0.55,
            "test_held_out_combination_ratio": 0.30,
            "held_out_combinations": [list(item) for item in HELD_OUT_COMBINATIONS],
            "held_out_test_examples": held_out_test_count,
        },
        "template_banks": {
            split: {
                "id_prefix": split,
                "unique_template_ids": len(
                    {
                        template_id
                        for item in metadata_by_split[split]
                        for template_id in item["template_ids"]
                    }
                ),
            }
            for split in ("train", "validation", "test")
        },
        "record_shapes": {
            split: {
                "single_constraint": sum(
                    len(record["label"]["constraints"]) == 1
                    for record in records_by_split[split]
                ),
                "multiple_constraints": sum(
                    len(record["label"]["constraints"]) > 1
                    for record in records_by_split[split]
                ),
                "team_wide": sum(
                    all(
                        "employee" not in constraint
                        for constraint in record["label"]["constraints"]
                    )
                    for record in records_by_split[split]
                ),
            }
            for split in ("train", "validation", "test")
        },
        **stats,
    }
    return records_by_split, manifest


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_preview(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    records, manifest = build_preview(seed)
    _atomic_jsonl(output_dir / "preview.jsonl", records)
    _atomic_json(output_dir / "preview_manifest.json", manifest)
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def verify_preview(output_dir: Path) -> None:
    preview_path = output_dir / "preview.jsonl"
    if not preview_path.exists():
        raise ValueError("Generate and review data/preview.jsonl before the full dataset")
    records = _load_jsonl(preview_path)
    if len(records) != 20:
        raise ValueError("The approved preview must contain exactly 20 records")
    validate_records(records)


def write_full_dataset(
    output_dir: Path,
    *,
    approved_preview: bool,
    seed: int = DEFAULT_SEED,
    sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not approved_preview:
        raise ValueError("Full generation requires explicit preview approval")
    verify_preview(output_dir)
    records_by_split, manifest = build_full_dataset(seed, sizes)
    for split, records in records_by_split.items():
        _atomic_jsonl(output_dir / SPLIT_FILENAMES[split], records)
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def validate_output_directory(output_dir: Path) -> dict[str, Any]:
    available = {
        split: _load_jsonl(output_dir / filename)
        for split, filename in SPLIT_FILENAMES.items()
        if (output_dir / filename).exists()
    }
    if available:
        return audit_splits(available)
    preview = _load_jsonl(output_dir / "preview.jsonl")
    validate_records(preview)
    return {"counts": {"preview": len(preview)}, "status": "valid"}
