"""Discover and match GRIB2 files for ensemble assembly."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from EnsDataStore.core.models import ForecastFile

CYCLE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
FHOUR_RE = re.compile(r"[Ff](\d{1,3})(?!\d)")
KNOWN_MEMBER_PREFIXES = ("HIRESW", "NAMNEST", "WRF4NSSL", "HRRR")


def discover_files(input_roots: Sequence[Path | str], patterns: Sequence[str], exclude_dirs: Sequence[str]) -> list[Path]:
    """Discover GRIB files across one or more directory roots."""
    files: list[Path] = []
    exclude_tokens = [str(item) for item in exclude_dirs]

    for root in input_roots:
        root_path = Path(root)
        for pattern in patterns:
            for path in root_path.rglob(pattern):
                if any(token in str(path) for token in exclude_tokens):
                    continue
                files.append(path)

    return files


def parse_filename(path: Path) -> tuple[str, datetime, int] | None:
    """Parse member, cycle time, and forecast hour from a GRIB filename/path."""
    name = path.name
    cycle_match = CYCLE_RE.search(name)
    forecast_hour_matches = list(FHOUR_RE.finditer(name))
    forecast_hour_match = forecast_hour_matches[-1] if forecast_hour_matches else None

    if not cycle_match or not forecast_hour_match:
        return None

    try:
        cycle_time = datetime.strptime(cycle_match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        forecast_hour = int(forecast_hour_match.group(1))
    except ValueError:
        return None

    member = _infer_member(path)
    return member, cycle_time, forecast_hour


def build_file_index(files: Sequence[Path]) -> list[ForecastFile]:
    """Build a typed index of forecast files with cycle and valid times."""
    index: list[ForecastFile] = []

    for path in files:
        parsed = parse_filename(path)
        if parsed is None:
            continue

        member, cycle_time, forecast_hour = parsed
        valid_time = cycle_time + timedelta(hours=forecast_hour)
        index.append(
            ForecastFile(
                path=path,
                member=member,
                name=path.name,
                cycle_time=cycle_time,
                valid_time=valid_time,
                forecast_hour=forecast_hour,
            )
        )

    return index


def match_by_valid_time(files: Sequence[ForecastFile], max_lags: int, cycle_spacing_hours: int = 6) -> dict[datetime, list[ForecastFile]]:
    """Group files by valid time, keeping cycles within the requested lag window."""
    if not files:
        return {}

    matches: dict[datetime, list[ForecastFile]] = defaultdict(list)
    cycles = sorted({forecast_file.cycle_time for forecast_file in files}, reverse=True)
    latest_cycle = cycles[0]
    max_allowed_lag_hours = max_lags * cycle_spacing_hours

    for forecast_file in files:
        lag_hours = int((latest_cycle - forecast_file.cycle_time).total_seconds() / 3600)
        if lag_hours <= max_allowed_lag_hours:
            matches[forecast_file.valid_time].append(forecast_file)

    return dict(matches)


def _infer_member(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith(KNOWN_MEMBER_PREFIXES):
            return part
    return path.parent.name
