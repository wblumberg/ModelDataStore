"""Build a single-member forecast Zarr store from local GRIB2 files."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import grib2io
import numpy as np
import zarr
from numcodecs import Blosc

if __package__ in (None, ""):
    # Allow running this file directly (python generate_memberStore.py) from src/EnsDataStore/pipelines.
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

try:
    from EnsDataStore.catalog.manifest import build_store_metadata
except ImportError:
    def build_store_metadata(
        *,
        product_type: str,
        system: str,
        run_id: str,
        source: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "product_type": product_type,
            "system": system,
            "run_id": run_id,
            "source": source,
        }
        if extra:
            payload.update(dict(extra))
        return payload

from EnsDataStore.core.models import ForecastFile, GridInfo, VariableInfo
from EnsDataStore.grib.discovery import build_file_index, discover_files, match_by_valid_time
from EnsDataStore.grib.grid import extract_grid_info
from EnsDataStore.grib.inventory import load_variables_db

try:
    from EnsDataStore.pipelines.adjustments import adjust_accumulated_for_lagged
except ImportError:
    def adjust_accumulated_for_lagged(file: ForecastFile, base_cycle: datetime, variables: Sequence[VariableInfo]) -> None:
        return None


def adjust_accumulated_for_lagged_file(file: ForecastFile, base_cycle: datetime, variables: Sequence[VariableInfo]) -> None:
    """Backward-compatible alias for the lagged accumulation hook."""
    adjust_accumulated_for_lagged(file, base_cycle, variables)


def write_member_zarr(
    matches: dict[datetime, list[ForecastFile]],
    variables: Sequence[VariableInfo],
    grid: GridInfo,
    output_path: Path,
    member: str,
    base_cycle: datetime,
    store_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a member's data into a Zarr store with xarray-compatible metadata."""
    logger = logging.getLogger(__name__)

    logger.info("Writing Zarr store for member %s to %s", member, output_path)

    times = sorted(matches.keys())
    ny, nx = grid.nj, grid.ni

    # Force Zarr V2 so numcodecs.Blosc and existing readers remain compatible.
    store = zarr.storage.LocalStore(str(output_path))
    root = zarr.group(store=store, overwrite=True, zarr_format=2)

    member_dtype = f"U{max(1, len(member))}"
    member_values = np.array([member], dtype=member_dtype)
    root.create_array("member", data=member_values)
    root["member"].attrs["_ARRAY_DIMENSIONS"] = ["member"]
    time_values = np.array([_to_np_datetime64_utc(value) for value in times], dtype="datetime64[ns]")
    root.create_array("time", data=time_values)
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    y_values = np.arange(ny, dtype=np.int32)
    root.create_array("y", data=y_values)
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    x_values = np.arange(nx, dtype=np.int32)
    root.create_array("x", data=x_values)
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

    root.attrs.update(
        {
            "member_name": member,
            "grid_type": "lambert",
            "ni": grid.ni,
            "nj": grid.nj,
            "lon_0": grid.lon_0,
            "lat_0": grid.lat_0,
            "lat_std": grid.lat_std,
            "ll_lat": grid.ll_lat,
            "ll_lon": grid.ll_lon,
            "ur_lat": grid.ur_lat,
            "ur_lon": grid.ur_lon,
            "description": f"Ensemble member {member} from CAM forecast data, stored in Zarr format.",
            "source": "Extracted from GRIB2 files using grib2io, processed with datastore.pipelines.member_store.",
            "history": f"Created on {_format_utc_timestamp(datetime.now(tz=timezone.utc))} by datastore.pipelines.member_store.",
            "forecast_times": len(times),
            "time_lagged": "",
        }
    )

    metadata = dict(store_metadata or {})
    metadata.setdefault("product_type", "member")
    metadata.setdefault("created_at", datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    root.attrs.update(metadata)

    compressor = Blosc(cname="zstd", clevel=5)
    variable_index = {
        _variable_signature(
            variable.grib_name,
            variable.level_type,
            variable.level1,
            variable.level2,
            variable.type,
        ): variable
        for variable in variables
    }

    short_name_index: dict[str, list[VariableInfo]] = {}
    for variable in variables:
        short_name_index.setdefault(variable.grib_name, []).append(variable)

    for variable in variables:
        logger.info("Creating variable %s", variable.name)
        accumulation_hours = _parse_accumulation_hours(variable.type)
        root.create_array(
            variable.name,
            shape=(len(times), 1, ny, nx),
            chunks=(1, 1, 353, 257),
            dtype=np.float32,
            fill_value=np.float32(np.nan),
            compressor=compressor,
        )
        root[variable.name].attrs.update(
            {
                "long_name": variable.long_name,
                "units": variable.units,
                "level1": variable.level1,
                "level2": variable.level2,
                "level_type": variable.level_type,
                "grib_name": variable.grib_name,
                "type": variable.type,
                "is_run_accumulated": bool(variable.type.startswith("accum")),
                "accumulation_hours": accumulation_hours,
                "_ARRAY_DIMENSIONS": ["time", "member", "y", "x"],
            }
        )

    total_assignments = 0
    source_cycles: list[np.datetime64] = []
    source_forecast_hours: list[int] = []
    source_lag_hours: list[int] = []
    for time_index, valid_time in enumerate(times):
        logger.info("Processing time %s/%s: %s", time_index + 1, len(times), valid_time)
        files = matches[valid_time]
        forecast_file = _select_source_file(files)
        logger.info("  Processing file %s", forecast_file.path)
        source_cycles.append(_to_np_datetime64_utc(forecast_file.cycle_time))
        source_forecast_hours.append(int(forecast_file.forecast_hour))
        lag_hours = int((base_cycle - forecast_file.cycle_time).total_seconds() // 3600)
        source_lag_hours.append(lag_hours)
        try:
            with grib2io.open(str(forecast_file.path)) as grib_file:
                for message in grib_file:
                    variable = _lookup_variable(message, variable_index, short_name_index)
                    if variable is None:
                        continue

                    root[variable.name][time_index, 0, :, :] = message.data
                    total_assignments += 1
        except Exception as exc:
            logger.error("Error reading %s: %s", forecast_file.path, exc)

    source_cycle_values = np.array(source_cycles, dtype="datetime64[ns]")
    root.create_array(
        "source_cycle_time",
        data=source_cycle_values,
    )
    root["source_cycle_time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    source_forecast_values = np.array(source_forecast_hours, dtype=np.int32)
    root.create_array(
        "source_forecast_hour",
        data=source_forecast_values,
    )
    root["source_forecast_hour"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    source_lag_values = np.array(source_lag_hours, dtype=np.int32)
    root.create_array(
        "source_lag_hours",
        data=source_lag_values,
    )
    root["source_lag_hours"].attrs["_ARRAY_DIMENSIONS"] = ["time"]

    logger.info("Completed %s assignments", total_assignments)
    zarr.consolidate_metadata(store)
    logger.info("Member Zarr store written with consolidated metadata")


def build_member_store(
    input_roots: Sequence[Path | str],
    patterns: Sequence[str],
    exclude_dirs: Sequence[str],
    member: str,
    variables_db: Path,
    output_dir: Path,
    cycle_time: datetime | None = None,
    latest_cycle_only: bool = False,
    max_lags: int = 0,
    max_times: int | None = None,
    cycle_spacing_hours: int = 6,
    store_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Build one member store end-to-end and return the output path."""
    logger = logging.getLogger(__name__)

    variables = load_variables_db(variables_db)
    files = discover_files(input_roots, patterns, exclude_dirs)
    logger.info("Discovered %s files", len(files))

    file_index = build_file_index(files)
    logger.info("Indexed %s files", len(file_index))

    member_files = [forecast_file for forecast_file in file_index if forecast_file.member == member]
    logger.info("Filtered to %s files for member %s", len(member_files), member)
    if not member_files:
        raise ValueError(f"No files for member {member}")

    if latest_cycle_only and cycle_time is not None:
        raise ValueError("--latest-cycle-only cannot be combined with --cycle-time")

    if latest_cycle_only:
        base_cycle = max(forecast_file.cycle_time for forecast_file in member_files)
        lagged_member_files = [forecast_file for forecast_file in member_files if forecast_file.cycle_time == base_cycle]
        logger.info("Using latest cycle only: %s", base_cycle.strftime("%Y%m%d%H"))
    else:
        base_cycle = _resolve_base_cycle(member_files, cycle_time)
        cycle_window_start = base_cycle - timedelta(hours=max_lags * cycle_spacing_hours)
        lagged_member_files = [
            forecast_file
            for forecast_file in member_files
            if cycle_window_start <= forecast_file.cycle_time <= base_cycle
        ]
        logger.info(
            "Using %s files between cycles %s and %s",
            len(lagged_member_files),
            cycle_window_start.strftime("%Y%m%d%H"),
            base_cycle.strftime("%Y%m%d%H"),
        )

    logger.info("Using base cycle %s", base_cycle.strftime("%Y%m%d%H"))
    if not lagged_member_files:
        raise ValueError(
            f"No files for member {member} within lag window for cycle {base_cycle.strftime('%Y%m%d%H')}"
        )

    matches = match_by_valid_time(lagged_member_files, max_lags, cycle_spacing_hours=cycle_spacing_hours)
    logger.info("Matched %s valid times", len(matches))
    if not matches:
        raise ValueError(f"No matched valid times for member {member} and cycle {base_cycle.strftime('%Y%m%d%H')}")

    if max_times:
        limited_times = sorted(matches.keys())[:max_times]
        matches = {valid_time: matches[valid_time] for valid_time in limited_times}
        logger.info("Limited to %s forecast times for debugging", len(matches))

    grid = extract_grid_info(member_files[0].path)
    logger.info("Grid: %sx%s", grid.ni, grid.nj)

    for matched_files in matches.values():
        for forecast_file in matched_files:
            if forecast_file.cycle_time < base_cycle:
                adjust_accumulated_for_lagged(forecast_file, base_cycle, variables)
    logger.info("Accumulated adjustment complete")

    system_name = Path(str(input_roots[0])).name if input_roots else "unknown"
    metadata = build_store_metadata(
        product_type="member",
        system=system_name,
        run_id=base_cycle.strftime("%Y%m%d%H"),
        source="datastore.pipelines.member_store",
        extra={"member": member},
    )
    if store_metadata:
        metadata.update(dict(store_metadata))

    output_path = Path(output_dir) / f"{member}.zarr"
    write_member_zarr(matches, variables, grid, output_path, member, base_cycle, store_metadata=metadata)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", nargs="+", required=True, help="Input root directories")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--member", required=True, help="Ensemble member name")
    parser.add_argument("--variables-db", required=True, help="Variables database JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for Zarr files")
    parser.add_argument("--cycle-time", help="Base cycle time in YYYYMMDDHH (UTC). Defaults to inferred current cycle.")
    parser.add_argument(
        "--latest-cycle-only",
        action="store_true",
        help="Use only files from the latest cycle found in input roots.",
    )
    parser.add_argument("--max-lags", type=int, default=0, help="Maximum lag hours")
    parser.add_argument("--max-times", type=int, help="Maximum number of forecast times to process")
    parser.add_argument("--cycle-spacing-hours", type=int, default=6, help="Cycle spacing in hours for lag windows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s]: %(message)s")
    build_member_store(
        input_roots=args.input_root,
        patterns=args.patterns,
        exclude_dirs=args.exclude_dirs,
        member=args.member,
        variables_db=Path(args.variables_db),
        output_dir=Path(args.output_dir),
        cycle_time=_parse_cycle_time(args.cycle_time) if args.cycle_time else None,
        latest_cycle_only=args.latest_cycle_only,
        max_lags=args.max_lags,
        max_times=args.max_times,
        cycle_spacing_hours=args.cycle_spacing_hours,
    )
    logging.getLogger(__name__).info("Member build complete")
    return 0


def _parse_cycle_time(raw_value: str) -> datetime:
    try:
        return datetime.strptime(raw_value, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Invalid --cycle-time {raw_value!r}. Expected YYYYMMDDHH in UTC") from exc


def _resolve_base_cycle(files: Sequence[ForecastFile], requested_cycle: datetime | None) -> datetime:
    available_cycles = sorted({forecast_file.cycle_time for forecast_file in files})
    if not available_cycles:
        raise ValueError("No available cycles for member")

    if requested_cycle is not None:
        if requested_cycle not in available_cycles:
            available_text = ", ".join(cycle.strftime("%Y%m%d%H") for cycle in available_cycles[-5:])
            raise ValueError(
                f"Requested cycle {requested_cycle.strftime('%Y%m%d%H')} not found. Recent available cycles: {available_text}"
            )
        return requested_cycle

    now_utc = datetime.now(tz=timezone.utc)
    not_future_cycles = [cycle for cycle in available_cycles if cycle <= now_utc]
    if not_future_cycles:
        return not_future_cycles[-1]
    return available_cycles[-1]


def _lookup_variable(
    message,
    variable_index: dict[tuple[str, str, str, str, str], VariableInfo],
    short_name_index: dict[str, list[VariableInfo]],
) -> VariableInfo | None:
    signature = _variable_signature(
        message.shortName,
        _level_type(message),
        _surface_value(message, "typeOfFirstFixedSurface", "valueOfFirstFixedSurface"),
        _surface_value(message, "typeOfSecondFixedSurface", "valueOfSecondFixedSurface"),
        _message_type(message),
    )
    variable = variable_index.get(signature)
    if variable is not None:
        return variable

    candidates = short_name_index.get(message.shortName, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _variable_signature(
    grib_name: str,
    level_type: str,
    level1: str,
    level2: str,
    variable_type: str,
) -> tuple[str, str, str, str, str]:
    return (grib_name, level_type, level1, level2, variable_type)


def _message_type(msg) -> str:
    if hasattr(msg, "statisticalProcess") and msg.statisticalProcess is not None:
        process_type = msg.statisticalProcess
        time_range = getattr(msg, "timeRangeOfStatisticalProcess", "")
        unit_of_time_range = getattr(msg, "unitOfTimeRangeOfStatisticalProcess", "")
        unit_of_time_range = unit_of_time_range.split("-")[1] if "-" in unit_of_time_range else unit_of_time_range

        if time_range and unit_of_time_range == "1":
            time_range += "h"

        if "Average" in process_type:
            return f"average_{time_range}h" if time_range else "average"
        if "Accumulation" in process_type:
            return f"accum_{time_range}h" if time_range else "accum"
        if "Maximum" in process_type:
            return f"max_{time_range}h" if time_range else "max"
        if "Minimum" in process_type:
            return f"min_{time_range}h" if time_range else "min"
        return f"stat_{process_type}_{time_range}" if time_range else f"stat_{process_type}"

    if hasattr(msg, "stepRange") and msg.stepRange:
        return f"accum_{msg.stepRange}"

    return "instant"


def _level_type(msg) -> str:
    surface_type = getattr(msg, "typeOfFirstFixedSurface", None)
    if surface_type == 103:
        return "HGHT"
    if surface_type == 100:
        return "PRES"
    if surface_type == 20:
        return "TEMP"
    return ""


def _surface_value(msg, surface_attr: str, value_attr: str) -> str:
    surface_type = getattr(msg, surface_attr, 255)
    if surface_type == 255:
        return ""
    if hasattr(msg, value_attr):
        return str(int(getattr(msg, value_attr)))
    return ""


def _to_np_datetime64_utc(value: datetime) -> np.datetime64:
    utc_value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(utc_value, "ns")


def _format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _select_source_file(files: Sequence[ForecastFile]) -> ForecastFile:
    if not files:
        raise ValueError("No source files available for valid time")
    return sorted(files, key=lambda file: (file.cycle_time, -file.forecast_hour), reverse=True)[0]


def _parse_accumulation_hours(variable_type: str) -> int:
    if not variable_type.startswith("accum"):
        return 0
    if variable_type == "accum":
        return 0

    suffix = variable_type.split("_", maxsplit=1)[1] if "_" in variable_type else ""
    if suffix.endswith("h"):
        suffix = suffix[:-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
