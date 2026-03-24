"""Variable inventory helpers for GRIB2-based model data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import grib2io

from EnsDataStore.core.models import VariableInfo


def get_variable_type(msg) -> str:
    """Determine the logical variable type from a GRIB2 message."""
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


def inventory_variables(files: Sequence[Path], sample_limit: int = 5) -> list[VariableInfo]:
    """Inventory variables from a sample of GRIB2 files.

    By default this keeps only variables that are present in all sampled files,
    which avoids hour-specific variants (for example accum_7h) polluting the
    inventory when they are not universally available.
    """
    return inventory_variables_with_presence(files, sample_limit=sample_limit, min_file_fraction=1.0)


def inventory_variables_with_presence(
    files: Sequence[Path],
    sample_limit: int = 5,
    min_file_fraction: float = 1.0,
) -> list[VariableInfo]:
    """Inventory variables and keep only those present in enough sampled files."""
    if min_file_fraction < 0.0 or min_file_fraction > 1.0:
        raise ValueError("min_file_fraction must be between 0.0 and 1.0")

    variables: dict[tuple[str, str, str, str, str], VariableInfo] = {}
    used_names: set[str] = set()
    file_presence_counts: dict[tuple[str, str, str, str, str], int] = {}
    processed_files = 0

    for path in list(files)[:sample_limit]:
        try:
            keys_in_file: set[tuple[str, str, str, str, str]] = set()
            with grib2io.open(str(path)) as grib_file:
                for msg in grib_file:
                    level_type = _level_type(msg)
                    level1 = _surface_value(msg, "typeOfFirstFixedSurface", "valueOfFirstFixedSurface")
                    level2 = _surface_value(msg, "typeOfSecondFixedSurface", "valueOfSecondFixedSurface")
                    variable_type = get_variable_type(msg)
                    key = (msg.shortName, level_type, level1, level2, variable_type)
                    keys_in_file.add(key)

                    if key in variables:
                        continue

                    variable_name = _build_variable_name(msg.shortName, level_type, level1, level2, variable_type, used_names)
                    variables[key] = VariableInfo(
                        name=variable_name,
                        grib_name=msg.shortName,
                        long_name=getattr(msg, "fullName", msg.shortName),
                        units=getattr(msg, "units", ""),
                        level_type=level_type,
                        level1=level1,
                        level2=level2,
                        type=variable_type,
                    )
            for key in keys_in_file:
                file_presence_counts[key] = file_presence_counts.get(key, 0) + 1
            processed_files += 1
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if processed_files == 0:
        return []

    retained_keys = [
        key
        for key, count in file_presence_counts.items()
        if count / processed_files >= min_file_fraction
    ]
    return [variables[key] for key in retained_keys]


def load_variables_db(variables_file: Path) -> list[VariableInfo]:
    with open(variables_file) as handle:
        payload = json.load(handle)
    return [VariableInfo.from_dict(item) for item in payload]


def save_variables_db(variables: Sequence[VariableInfo], output_file: Path) -> None:
    payload = [variable.to_dict() for variable in variables]
    with open(output_file, "w") as handle:
        json.dump(payload, handle, indent=2)


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


def _build_variable_name(
    short_name: str,
    level_type: str,
    level1: str,
    level2: str,
    variable_type: str,
    used_names: set[str],
) -> str:
    parts = [short_name]
    if level_type:
        if level1 and level2:
            parts.append(f"{level_type.lower()}_{level1}_{level2}")
        elif level1:
            parts.append(f"{level_type.lower()}_{level1}")
    if variable_type and variable_type != "instant":
        parts.append(variable_type)

    candidate = _sanitize_name("_".join(parts))
    name = candidate
    suffix = 2
    while name in used_names:
        name = f"{candidate}_{suffix}"
        suffix += 1

    used_names.add(name)
    return name


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "variable"
