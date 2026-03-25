"""Grid metadata extraction helpers for GRIB2 files."""

from __future__ import annotations

from pathlib import Path

import grib2io

from EnsDataStore.core.models import GridInfo


def _normalize_longitude_degrees(value: float) -> float:
    """Normalize longitude in degrees to the [-180, 180) convention."""
    return ((float(value) + 180.0) % 360.0) - 180.0


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

    # grib2io returns coordinates already in degrees — no /1e6 needed.
    # Use the 2-D lats/lons arrays for corner points when available.
    lats = getattr(message, "lats", None)
    lons = getattr(message, "lons", None)
    if lats is not None and lons is not None:
        ll_lat = float(lats[0, 0])
        ll_lon = float(lons[0, 0])
        ur_lat = float(lats[-1, -1])
        ur_lon = float(lons[-1, -1])
    else:
        ll_lat = getattr(message, "latitudeFirstGridpoint", 0.0)
        ll_lon = getattr(message, "longitudeFirstGridpoint", 0.0)
        ur_lat = 0.0
        ur_lon = 0.0

    return GridInfo(
        ni=ni or 0,
        nj=nj or 0,
        lat_0=getattr(message, "latitudeTrueScale", 0.0),
        lon_0=_normalize_longitude_degrees(getattr(message, "gridOrientation", 0.0)),
        lat_std=getattr(message, "standardLatitude1", 0.0),
        ll_lat=ll_lat,
        ll_lon=ll_lon,
        ur_lat=ur_lat,
        ur_lon=ur_lon,
    )
