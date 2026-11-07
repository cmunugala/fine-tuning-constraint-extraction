"""Command-line interface for synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    DEFAULT_SEED,
    validate_output_directory,
    write_full_dataset,
    write_preview,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="write the 20-row review dataset")
    preview.add_argument("--seed", type=int, default=DEFAULT_SEED)
    preview.add_argument("--output-dir", type=Path, default=Path("data"))

    generate = subparsers.add_parser("generate", help="write train/validation/test splits")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--output-dir", type=Path, default=Path("data"))
    generate.add_argument("--approved-preview", action="store_true")

    validate = subparsers.add_parser("validate", help="validate generated JSONL files")
    validate.add_argument("--input-dir", type=Path, default=Path("data"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preview":
        result = write_preview(args.output_dir, args.seed)
    elif args.command == "generate":
        result = write_full_dataset(
            args.output_dir,
            approved_preview=args.approved_preview,
            seed=args.seed,
        )
    else:
        result = validate_output_directory(args.input_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
