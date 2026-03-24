"""Build a postprocessed ensemble dataset from member Zarr stores."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import xarray as xr

if __package__ in (None, ""):
	# Allow running this file directly (python postprocess_ensemble.py) from src/EnsDataStore/pipelines.
	src_root = Path(__file__).resolve().parents[2]
	if str(src_root) not in sys.path:
		sys.path.insert(0, str(src_root))

from EnsDataStore.calc.accumulation import interval_accumulation, interval_to_run_accumulated


def discover_member_stores(input_roots: Sequence[Path | str], pattern: str = "*.zarr") -> list[Path]:
	stores: list[Path] = []
	for root in input_roots:
		root_path = Path(root)
		stores.extend(sorted(root_path.glob(pattern)))
	return stores


def open_ensemble_from_member_stores(
	store_paths: Sequence[Path],
	*,
	concat_join: str = "inner",
) -> xr.Dataset:
	if not store_paths:
		raise ValueError("No member stores found")

	datasets: list[xr.Dataset] = []
	for path in store_paths:
		ds = xr.open_zarr(str(path), consolidated=True)
		datasets.append(ds)

	ensemble = xr.concat(
		datasets,
		dim="member",
		join=concat_join,
		data_vars="all",
		coords="minimal",
		compat="override",
		combine_attrs="override",
	)
	if "time" in ensemble.coords:
		ensemble = ensemble.sortby("time")
	return ensemble


def add_accumulation_products(
	ds: xr.Dataset,
	*,
	accumulation_grib_names: Sequence[str],
	prefer_native_1h: bool = True,
	reconstruct_run_total: bool = True,
) -> xr.Dataset:
	out = ds.copy()
	source_cycle = out["source_cycle_time"] if "source_cycle_time" in out else None
	source_forecast_hour = out["source_forecast_hour"] if "source_forecast_hour" in out else None

	for grib_name in accumulation_grib_names:
		candidates = _accumulation_candidates(out, grib_name)
		if not candidates:
			continue

		one_hour_name = _first_name_with_type(candidates, "accum_1h")
		run_name = _choose_run_accumulation_name(candidates)

		one_hour_da = out[one_hour_name] if one_hour_name else None
		run_da = out[run_name] if run_name else None

		interval_da = interval_accumulation(
			one_hour_accumulated=one_hour_da if prefer_native_1h else None,
			run_accumulated=run_da if (not prefer_native_1h or one_hour_da is None) else None,
			source_cycle_var=source_cycle,
			source_forecast_hour_var=source_forecast_hour,
		)

		interval_name = f"{grib_name}_interval_1h"
		interval_da.name = interval_name
		interval_da.attrs["grib_name"] = grib_name
		out[interval_name] = interval_da

		if reconstruct_run_total:
			run_total_name = f"{grib_name}_run_total"
			run_total = interval_to_run_accumulated(
				out[interval_name],
				source_cycle_var=source_cycle,
				source_forecast_hour_var=source_forecast_hour,
			)
			run_total.name = run_total_name
			run_total.attrs["grib_name"] = grib_name
			out[run_total_name] = run_total

	return out


def write_postprocessed_ensemble(ds: xr.Dataset, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	ds.to_zarr(str(output_path), mode="w", consolidated=True, zarr_format=2)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-root", nargs="+", required=True, help="Directory roots containing member .zarr stores")
	parser.add_argument("--member-pattern", default="*.zarr", help="Glob pattern for member stores")
	parser.add_argument("--output-path", required=True, help="Output ensemble Zarr path")
	parser.add_argument(
		"--concat-join",
		choices=("inner", "outer", "left", "right", "exact", "override"),
		default="inner",
		help="Join strategy when concatenating members by time",
	)
	parser.add_argument(
		"--accum-grib-name",
		nargs="*",
		default=["APCP", "WEASD"],
		help="GRIB short names to derive accumulation products for",
	)
	parser.add_argument(
		"--prefer-native-1h",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Prefer native accum_1h fields when available (default: true)",
	)
	parser.add_argument(
		"--reconstruct-run-total",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Also reconstruct cycle-reset run totals from interval accumulations (default: true)",
	)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s]: %(message)s")
	logger = logging.getLogger(__name__)

	store_paths = discover_member_stores(args.input_root, pattern=args.member_pattern)
	logger.info("Discovered %s member stores", len(store_paths))
	if not store_paths:
		raise ValueError("No member stores discovered")

	ensemble = open_ensemble_from_member_stores(store_paths, concat_join=args.concat_join)
	logger.info("Loaded ensemble with dims %s", dict(ensemble.sizes))

	postprocessed = add_accumulation_products(
		ensemble,
		accumulation_grib_names=args.accum_grib_name,
		prefer_native_1h=args.prefer_native_1h,
		reconstruct_run_total=args.reconstruct_run_total,
	)
	logger.info("Postprocessed ensemble has variables: %s", len(postprocessed.data_vars))

	write_postprocessed_ensemble(postprocessed, Path(args.output_path))
	logger.info("Wrote postprocessed ensemble to %s", args.output_path)
	return 0


def _accumulation_candidates(ds: xr.Dataset, grib_name: str) -> dict[str, str]:
	candidates: dict[str, str] = {}
	for name, variable in ds.data_vars.items():
		var_grib_name = variable.attrs.get("grib_name")
		var_type = str(variable.attrs.get("type", ""))
		if var_grib_name == grib_name and var_type.startswith("accum"):
			candidates[name] = var_type
	return candidates


def _first_name_with_type(candidates: dict[str, str], wanted_type: str) -> str | None:
	for name, var_type in candidates.items():
		if var_type == wanted_type:
			return name
	return None


def _choose_run_accumulation_name(candidates: dict[str, str]) -> str | None:
	for name, var_type in candidates.items():
		if var_type == "accum":
			return name

	weighted: list[tuple[int, str]] = []
	for name, var_type in candidates.items():
		if var_type.startswith("accum") and var_type != "accum_1h":
			weighted.append((_accum_hours(var_type), name))
	if not weighted:
		return None
	weighted.sort(reverse=True)
	return weighted[0][1]


def _accum_hours(var_type: str) -> int:
	if var_type == "accum":
		return 0
	suffix = var_type.split("_", maxsplit=1)[1] if "_" in var_type else ""
	if suffix.endswith("h"):
		suffix = suffix[:-1]
	try:
		return int(suffix)
	except ValueError:
		return 0


if __name__ == "__main__":
	raise SystemExit(main())
