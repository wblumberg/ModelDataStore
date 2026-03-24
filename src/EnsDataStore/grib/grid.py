"""Grid metadata extraction helpers for GRIB2 files."""

from __future__ import annotations

from pathlib import Path

import grib2io

from EnsDataStore.core.models import GridInfo


def extract_grid_info(path: Path) -> GridInfo:
    """Extract grid information from the first suitable GRIB2 message."""
    message = None
    ni = None
    nj = None

    with grib2io.open(str(path)) as grib_file:
        for candidate in grib_file:
            message = candidate
            ni = getattr(candidate, "Ni", None) or getattr(candidate, "Nx", None)
            nj = getattr(candidate, "Nj", None) or getattr(candidate, "Ny", None)

            if ni is None or nj is None or ni == 0 or nj == 0:
                try:
                    data = candidate.data
                    if hasattr(data, "shape") and len(data.shape) == 2:
                        nj_data, ni_data = data.shape
                        ni = ni or ni_data
                        nj = nj or nj_data
                except Exception:
                    continue

            if ni and nj and ni > 0 and nj > 0:
                break

    if message is None:
        raise ValueError(f"No readable GRIB2 messages found in {path}")

    return GridInfo(
        ni=ni or 0,
        nj=nj or 0,
        lat_0=getattr(message, "LaD", 0) / 1e6 if hasattr(message, "LaD") else 0,
        lon_0=getattr(message, "LoV", 0) / 1e6 if hasattr(message, "LoV") else 0,
        lat_std=getattr(message, "Latin1", 0) / 1e6 if hasattr(message, "Latin1") else 0,
        ll_lat=getattr(message, "latitudeOfFirstGridPoint", 0) / 1e6,
        ll_lon=getattr(message, "longitudeOfFirstGridPoint", 0) / 1e6,
        ur_lat=getattr(message, "latitudeOfLastGridPoint", 0) / 1e6,
        ur_lon=getattr(message, "longitudeOfLastGridPoint", 0) / 1e6,
    )
