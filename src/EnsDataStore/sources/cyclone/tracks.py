"""Generate ATCF-like track files from ECMWF ensemble tropical cyclone forecasts."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np


def ms_to_knots(ms: float | None) -> int:
    if ms is None or np.isnan(ms):
        return 0
    return int(round(float(ms) * 1.94384))


def format_lat(lat: float | None) -> str:
    try:
        numeric = float(lat)
    except Exception:
        return "000N"
    if np.isnan(numeric):
        return "000N"
    hemi = "N" if numeric >= 0 else "S"
    val = int(round(abs(numeric) * 10))
    return f"{val:03d}{hemi}"


def format_lon(lon: float | None) -> str:
    try:
        numeric = float(lon)
    except Exception:
        return "0000E"
    if np.isnan(numeric):
        return "0000E"
    hemi = "E" if numeric >= 0 else "W"
    val = int(round(abs(numeric) * 10))
    return f"{val:04d}{hemi}"


def get_basin_code2(sid: str) -> str:
    if "L" in sid:
        return "AL"
    if "W" in sid:
        return "WP"
    if "C" in sid:
        return "CP"
    if "E" in sid:
        return "EP"
    if "P" in sid:
        return "SP"
    if "A" in sid:
        return "IO"
    if "B" in sid:
        return "BB"
    if "U" in sid:
        return "AU"
    if "S" in sid:
        return "SI"
    if "X" in sid:
        return "XX"
    return "XX"


def get_tc_type(category: str) -> str:
    cat = str(category).lower()
    if "tropical storm" in cat:
        return "TS"
    if "typhoon" in cat:
        return "TY"
    if "hurricane" in cat:
        return "HU"
    if "tropical depression" in cat:
        return "TD"
    if "extratropical" in cat:
        return "EX"
    if "post" in cat:
        return "PT"
    if "subtropical" in cat:
        return "SS"
    return "XX"


def to_np_datetime(value) -> np.datetime64 | None:
    if value is None:
        return None
    try:
        return np.datetime64(value)
    except Exception:
        try:
            return np.datetime64(str(value))
        except Exception:
            return None


def _determine_output_dtg(tc_forecast) -> str:
    dtg = None
    try:
        if tc_forecast.data:
            first_ds = tc_forecast.data[0]
            attrs = getattr(first_ds, "attrs", {}) or {}
            run_dt = to_np_datetime(attrs.get("run_datetime"))
            if run_dt is None:
                try:
                    run_dt = first_ds["time"].isel(time=0).values
                except Exception:
                    run_dt = None
            if run_dt is not None:
                dtg = str(run_dt).replace("-", "").replace(":", "").replace("T", "")[:10]
    except Exception:
        dtg = None

    if dtg is None:
        now = np.datetime64("now").astype("datetime64[h]")
        dtg = str(now).replace("-", "").replace(":", "").replace("T", "")[:10]
    return dtg


def build_ecmwf_tracks(*, out_dir: Path | str = "/data/gempak/storm/enstrack/") -> Path:
    from climada_petals.hazard import TCForecast

    tc_forecast = TCForecast()
    tc_forecast.fetch_ecmwf()

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dtg = _determine_output_dtg(tc_forecast)
    target = out_path / f"cyclone_{dtg}"

    with open(target, "w", encoding="utf-8") as handle:
        for ds in tc_forecast.data:
            try:
                attrs = ds.attrs
                sid = str(attrs.get("sid", "01")).strip()
                cyclone_num = "".join(filter(str.isdigit, sid)) or "01"
                basin = get_basin_code2(sid)

                run_dt = to_np_datetime(attrs.get("run_datetime"))
                if run_dt is None:
                    run_dt = ds["time"].isel(time=0).values

                dtg_local = str(run_dt).replace("-", "").replace(":", "").replace("T", "")[:10]

                technum = int(attrs.get("ensemble_number", 1)) - 1
                technum = max(0, technum)
                tech = "EC00" if technum == 0 else f"EP{technum:02d}"

                category = attrs.get("category", "XX")
                n_times = ds["time"].sizes.get("time", None) or len(ds["time"])

                for idx in range(n_times):
                    time_i = ds["time"].isel(time=idx).values
                    try:
                        tau = int((np.datetime64(time_i) - np.datetime64(run_dt)) / np.timedelta64(1, "h"))
                    except Exception:
                        tau = idx * 6

                    try:
                        lat_i = ds["lat"].isel(time=idx).item()
                    except Exception:
                        lat_i = np.nan

                    try:
                        lon_i = ds["lon"].isel(time=idx).item()
                    except Exception:
                        lon_i = np.nan

                    try:
                        wind_i = ds["max_sustained_wind"].isel(time=idx).item()
                    except Exception:
                        wind_i = np.nan

                    try:
                        pres_i = ds["central_pressure"].isel(time=idx).item()
                    except Exception:
                        pres_i = np.nan

                    vmax = ms_to_knots(wind_i)
                    mslp = int(round(pres_i)) if not np.isnan(pres_i) else 0
                    tc_type = get_tc_type(category)

                    row = [
                        basin,
                        cyclone_num.zfill(2),
                        dtg_local,
                        f"{technum:02d}",
                        tech,
                        f"{int(tau):3d}",
                        format_lat(lat_i),
                        format_lon(lon_i),
                        f"{vmax:3d}",
                        f"{mslp:5d}",
                        tc_type,
                        " 34",
                        "AAA",
                        "  0",
                        "  0",
                        "  0",
                        "  0",
                        "   0",
                        "   0",
                        "  0",
                        "  0",
                        "  0",
                        "  0",
                    ]
                    handle.write(",".join(str(item) for item in row) + "\n")
            finally:
                close_fn = getattr(ds, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                gc.collect()

    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ATCF-like track file from ECMWF ensemble data")
    parser.add_argument("--out-dir", default="/data/gempak/storm/enstrack/", help="Output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target = build_ecmwf_tracks(out_dir=args.out_dir)
    print(f"Wrote {target}")
    return 0


__all__ = [
    "build_ecmwf_tracks",
    "build_parser",
    "format_lat",
    "format_lon",
    "get_basin_code2",
    "get_tc_type",
    "main",
    "ms_to_knots",
]


if __name__ == "__main__":
    raise SystemExit(main())
