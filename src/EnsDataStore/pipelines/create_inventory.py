"""Build a reusable JSON variable inventory from GRIB2 files.

   The inventory consists of the GRIB2 messages that are consistent
   across all of the GRIB2 files that are in a given directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    # Allow running this file directly (python create_inventory.py) from src/EnsDataStore/pipelines.
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from EnsDataStore.grib.discovery import discover_files
from EnsDataStore.grib.inventory import inventory_variables_with_presence, save_variables_db


def inventory_variables_database(
    input_roots: Sequence[Path | str],
    patterns: Sequence[str],
    exclude_dirs: Sequence[str],
    output_variables: Path,
    sample_limit: int = 30,
    min_file_fraction: float = 1.0,
) -> list:
    files = discover_files(input_roots, patterns, exclude_dirs)
    sample_files = files[:sample_limit]
    variables = inventory_variables_with_presence(
        sample_files,
        sample_limit=sample_limit,
        min_file_fraction=min_file_fraction,
    )
    save_variables_db(variables, output_variables)
    return variables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", nargs="+", required=True, help="Input root directories")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--output-variables", required=True, help="Output JSON file for variables")
    parser.add_argument("--sample-limit", type=int, default=30, help="Maximum sample files to inventory")
    parser.add_argument(
        "--min-file-fraction",
        type=float,
        default=1.0,
        help=(
            "Keep only variables present in at least this fraction of sampled files "
            "(0.0-1.0). Default 1.0 keeps variables common to all sampled files."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output_variables)
    variables = inventory_variables_database(
        input_roots=args.input_root,
        patterns=args.patterns,
        exclude_dirs=args.exclude_dirs,
        output_variables=output_path,
        sample_limit=args.sample_limit,
        min_file_fraction=args.min_file_fraction,
    )
    print(f"Inventory complete. {len(variables)} variables saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
