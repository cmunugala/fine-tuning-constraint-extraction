"""Dataset orchestration, review gating, and atomic JSONL output."""

from __future__ import annotations

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
    PAIR_COMBINATIONS,
    TRIPLE_COMBINATIONS,
    combination_key,
)
from .generation import Scenario, make_scenario
from .rendering import render_scenario
from .validation import (
    audit_splits,
    canonical_label,
    fingerprint,
    normalized_text,
    validate_record,
    validate_records,
)

DEFAULT_SEED = 20261102
DEFAULT_SIZES = {"train": 1000, "validation": 175, "test": 175}
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}


def _record(
    scenario: Scenario, rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]]:
    text, render_ids, style = render_scenario(scenario, rng)
    label = scenario.output.model_dump(mode="json", exclude_none=True)
    record = {
        "text": text,
        "reference_date": scenario.reference_date.isoformat(),
        "label": label,
    }
    validate_record(record)
    metadata = {
        "render_ids": list(render_ids),
        "style": style,
        "combination": list(combination_key([item.type for item in scenario.output.constraints])),
    }
    return record, metadata


def _cycled_plans(
    combinations: tuple[tuple[str, ...], ...], count: int
) -> list[tuple[str, ...]]:
    return [combinations[index % len(combinations)] for index in range(count)]


def _plans_for_pool(size: int, rng: random.Random) -> list[tuple[str, ...]]:
    single_count = math.floor(size * 0.55 + 0.5)
    double_count = math.floor(size * 0.27 + 0.5)
    triple_count = size - single_count - double_count
    singles = tuple((constraint_type,) for constraint_type in CONSTRAINT_TYPES)
    plans = _cycled_plans(singles, single_count)
    plans.extend(_cycled_plans(PAIR_COMBINATIONS, double_count))
    plans.extend(_cycled_plans(TRIPLE_COMBINATIONS, triple_count))
    rng.shuffle(plans)
    return plans


def _generate_pool(
    size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    plans = _plans_for_pool(size, rng)
    used_texts: set[str] = set()
    used_labels: set[str] = set()

    for types in plans:
        accepted = False
        for _ in range(500):
            team_wide = rng.random() < 0.10
            multiple_employees = (
                len(types) > 1 and not team_wide and rng.random() < 0.15
            )
            scenario = make_scenario(
                types,
                rng,
                team_wide=team_wide,
                multiple_employees=multiple_employees,
            )
            record, item_metadata = _record(scenario, rng)
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
            raise RuntimeError(f"Could not generate a unique pooled record for {types}")
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
        record, item_metadata = _record(scenario, rng)
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
        "unique_render_components": len(
            {render_id for item in metadata for render_id in item["render_ids"]}
        ),
        **stats,
    }


def build_full_dataset(
    seed: int = DEFAULT_SEED,
    sizes: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sizes = dict(sizes or DEFAULT_SIZES)
    if set(sizes) != {"train", "validation", "test"}:
        raise ValueError("sizes must define train, validation, and test")
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("split sizes must be positive")

    pool_size = sum(sizes.values())
    records, metadata = _generate_pool(pool_size, random.Random(seed))
    indices = list(range(pool_size))
    split_seed = seed ^ 0x5DEECE66D
    random.Random(split_seed).shuffle(indices)

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    metadata_by_split: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        selected = indices[offset : offset + sizes[split]]
        records_by_split[split] = [records[index] for index in selected]
        metadata_by_split[split] = [metadata[index] for index in selected]
        offset += sizes[split]

    stats = audit_splits(records_by_split, metadata_by_split)
    if pool_size >= 1000 and stats["unique_text_skeletons"] < 200:
        raise RuntimeError("Generated pool has fewer than 200 text skeletons")
    if pool_size >= 1000 and stats["max_text_skeleton_share"] > 0.02:
        raise RuntimeError("A text skeleton exceeds 2% of the generated pool")

    manifest = {
        "dataset": "scheduling_constraints",
        "seed": seed,
        "pool_size": pool_size,
        "schema_model": "models.SchedulingOutput",
        "row_fields": ["text", "reference_date", "label"],
        "null_fields_omitted": True,
        "split_policy": {
            "method": "seeded_random_shuffle",
            "stratified": False,
            "split_seed": split_seed,
            "single_constraint_ratio": 0.55,
            "double_constraint_ratio": 0.27,
            "triple_constraint_ratio": 0.18,
            "shared_language_distribution": True,
        },
        "unique_render_components": len(
            {render_id for item in metadata for render_id in item["render_ids"]}
        ),
        "constraint_combinations": len(
            {tuple(item["combination"]) for item in metadata}
        ),
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
