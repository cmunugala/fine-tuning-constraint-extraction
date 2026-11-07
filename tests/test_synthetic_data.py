from __future__ import annotations

from datetime import date, time
from pathlib import Path
import tempfile
import unittest

from models import SchedulingConstraint, SchedulingOutput, TimeScope
from synthetic_data.catalog import HELD_OUT_KEYS, combination_key
from synthetic_data.pipeline import (
    build_full_dataset,
    build_preview,
    write_full_dataset,
    write_preview,
)
from synthetic_data.generation import make_scenario
from synthetic_data.rendering import assert_template_banks_disjoint, render_scenario
from synthetic_data.validation import audit_splits, validate_output


class SyntheticDataTests(unittest.TestCase):
    def test_template_banks_are_disjoint(self) -> None:
        assert_template_banks_disjoint()

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
        self.assertEqual(len(types), 7)
        self.assertTrue(any(len(row["label"]["constraints"]) > 1 for row in first))
        self.assertTrue(
            any("employee" not in item for row in first for item in row["label"]["constraints"])
        )
        self.assertTrue(any("tomorrow" in row["text"].lower() for row in first))

    def test_small_full_dataset_has_no_leakage(self) -> None:
        sizes = {"train": 40, "validation": 20, "test": 20}
        records, manifest = build_full_dataset(456, sizes)
        self.assertEqual(manifest["counts"], sizes)
        self.assertEqual(audit_splits(records)["counts"], sizes)
        for split in ("train", "validation"):
            for row in records[split]:
                key = combination_key(
                    [item["type"] for item in row["label"]["constraints"]]
                )
                self.assertNotIn(key, HELD_OUT_KEYS)
        self.assertTrue(
            any(
                combination_key([item["type"] for item in row["label"]["constraints"]])
                in HELD_OUT_KEYS
                for row in records["test"]
            )
        )

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

    def test_relative_dates_cross_year_and_leap_day(self) -> None:
        import random

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
            text, _ = render_scenario(scenario, "preview", random.Random(8))
            self.assertIn("tomorrow", text.lower())
            self.assertEqual(
                scenario.output.constraints[0].time_scope.start_date,
                expected_date,
            )

    def test_audit_rejects_duplicate_and_held_out_train_records(self) -> None:
        records, _ = build_preview(321)
        with self.assertRaisesRegex(ValueError, "Duplicate request"):
            audit_splits({"train": [records[0]], "validation": [records[0]]})

        held_out = next(
            record
            for record in records
            if combination_key(
                [item["type"] for item in record["label"]["constraints"]]
            )
            in HELD_OUT_KEYS
        )
        with self.assertRaisesRegex(ValueError, "Held-out combination"):
            audit_splits({"train": [held_out]})

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
