from __future__ import annotations

from datetime import date, time
from pathlib import Path
import random
import tempfile
import unittest

from models import SchedulingConstraint, SchedulingOutput, TimeScope
from synthetic_data.catalog import CONSTRAINT_TYPES
from synthetic_data.generation import make_scenario
from synthetic_data.pipeline import (
    DEFAULT_SIZES,
    build_full_dataset,
    build_preview,
    write_full_dataset,
    write_preview,
)
from synthetic_data.rendering import STYLES, render_scenario
from synthetic_data.validation import audit_splits, validate_output


class SyntheticDataTests(unittest.TestCase):
    def test_preview_is_deterministic_and_representative(self) -> None:
        first, manifest = build_preview(123)
        second, _ = build_preview(123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(manifest["counts"], {"preview": 20})
        types = {
            constraint["type"]
            for record in first
            for constraint in record["label"]["constraints"]
        }
        self.assertEqual(types, set(CONSTRAINT_TYPES))
        self.assertTrue(any(len(row["label"]["constraints"]) > 1 for row in first))
        self.assertTrue(
            any(
                "employee" not in item
                for row in first
                for item in row["label"]["constraints"]
            )
        )

    def test_pool_is_generated_before_seeded_random_split(self) -> None:
        sizes = {"train": 40, "validation": 20, "test": 20}
        first, manifest = build_full_dataset(456, sizes)
        second, second_manifest = build_full_dataset(456, sizes)
        different, _ = build_full_dataset(457, sizes)

        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        self.assertNotEqual(first, different)
        self.assertEqual(manifest["counts"], sizes)
        self.assertEqual(manifest["pool_size"], 80)
        self.assertEqual(manifest["split_policy"]["method"], "seeded_random_shuffle")
        self.assertFalse(manifest["split_policy"]["stratified"])
        self.assertTrue(manifest["split_policy"]["shared_language_distribution"])
        self.assertEqual(audit_splits(first)["unique_texts"], 80)
        self.assertEqual(audit_splits(first)["unique_labels"], 80)

    def test_full_default_dataset_meets_diversity_requirements(self) -> None:
        records, manifest = build_full_dataset(789)
        self.assertEqual(manifest["counts"], DEFAULT_SIZES)
        self.assertEqual(manifest["pool_size"], 1350)
        self.assertGreaterEqual(manifest["unique_text_skeletons"], 200)
        self.assertLessEqual(manifest["max_text_skeleton_share"], 0.02)
        self.assertEqual(manifest["constraint_combinations"], 63)
        self.assertEqual(set(manifest["render_style_counts"]), set(STYLES))
        self.assertEqual(set(manifest["constraint_priority_counts"]), {"hard", "soft"})

        for rows in records.values():
            for row in rows:
                for constraint in row["label"]["constraints"]:
                    if constraint["type"] == "available":
                        self.assertEqual(constraint["priority"], "hard")
                    if constraint["type"] == "preferred_hours":
                        self.assertEqual(constraint["priority"], "soft")

    def test_semantic_validation_rejects_reversed_times(self) -> None:
        output = SchedulingOutput(
            constraints=[
                SchedulingConstraint(
                    employee="Sam",
                    type="available",
                    time_scope=TimeScope(
                        start_date=date(2026, 1, 5),
                        end_date=date(2026, 1, 5),
                        start_time=time(17),
                        end_time=time(9),
                    ),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "start_time"):
            validate_output(output)

    def test_availability_and_preference_semantics_are_enforced(self) -> None:
        scope = TimeScope(days_of_week=["monday"], recurring=True)
        with self.assertRaisesRegex(ValueError, "available constraints must be hard"):
            validate_output(
                SchedulingOutput(
                    constraints=[
                        SchedulingConstraint(
                            employee="Sam",
                            type="available",
                            priority="soft",
                            time_scope=scope,
                        )
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "preferred_hours constraints must be soft"):
            validate_output(
                SchedulingOutput(
                    constraints=[
                        SchedulingConstraint(
                            employee="Sam",
                            type="preferred_hours",
                            priority="hard",
                            time_scope=scope,
                        )
                    ]
                )
            )

    def test_relative_dates_cross_year_and_leap_day(self) -> None:
        cases = (
            (date(2026, 12, 31), date(2027, 1, 1)),
            (date(2028, 2, 28), date(2028, 2, 29)),
        )
        for reference_date, expected_date in cases:
            scenario = make_scenario(
                ("available",),
                random.Random(7),
                reference_date=reference_date,
                scope_styles=("one_off",),
                priorities=("hard",),
                one_off_offsets=(1,),
            )
            text, _, _ = render_scenario(scenario, random.Random(8))
            self.assertTrue(text)
            self.assertEqual(
                scenario.output.constraints[0].time_scope.start_date,
                expected_date,
            )

    def test_audit_rejects_duplicates(self) -> None:
        records, _ = build_preview(321)
        with self.assertRaisesRegex(ValueError, "Duplicate request"):
            audit_splits({"train": [records[0]], "validation": [records[0]]})

    def test_full_write_requires_approval_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "approval"):
                write_full_dataset(
                    output,
                    approved_preview=False,
                    sizes={"train": 4, "validation": 2, "test": 2},
                )
            with self.assertRaisesRegex(ValueError, "preview"):
                write_full_dataset(
                    output,
                    approved_preview=True,
                    sizes={"train": 4, "validation": 2, "test": 2},
                )

    def test_full_write_uses_requested_split_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_preview(output, seed=12)
            write_full_dataset(
                output,
                approved_preview=True,
                seed=12,
                sizes={"train": 10, "validation": 4, "test": 4},
            )
            self.assertTrue((output / "train.jsonl").is_file())
            self.assertTrue((output / "val.jsonl").is_file())
            self.assertTrue((output / "test.jsonl").is_file())
            self.assertFalse((output / "validation.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
