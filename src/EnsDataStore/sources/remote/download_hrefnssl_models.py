"""Download HREF/CAM ensemble GRIB2 members from NSSL THREDDS."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

MODELS = [
    {
        "outdir": "/home/gblumberg/data/gempak/model/hiresw",
        "gribdir": "/home/gblumberg/data/base/model/hiresw_conusfv3",
        "model": "hiresw",
        "product": "fv3_2p5km",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusfv3_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hiresw_conusfv3.gem",
        "model_ext": "_hiresw_conusfv3.gem",
        "fxx": range(0, 61, 1),
        "params": None,
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/hiresw",
        "gribdir": "/home/gblumberg/data/base/model/hiresw_conusarw",
        "model": "hiresw",
        "product": "arw_2p5km",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusarw_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hiresw_conusarw.gem",
        "model_ext": "_hires_conusarw.gem",
        "fxx": range(0, 49, 1),
        "params": None,
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/namnest",
        "gribdir": "/home/gblumberg/data/base/model/namnest",
        "model": "nam",
        "product": "conusnest.hiresf",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/nam_conusnest_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_namnest.gem",
        "model_ext": "_namnest.gem",
        "fxx": range(0, 61, 1),
        "params": None,
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/wrf4nssl",
        "gribdir": "/home/gblumberg/data/base/model/wrf4nssl",
        "model": "wrf4nssl",
        "product": None,
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusnssl_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_wrf4nssl.gem",
        "model_ext": "_nsslwrf.gem",
        "fxx": range(0, 49, 1),
        "params": None,
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/hrrr",
        "gribdir": "/home/gblumberg/data/base/model/hrrr",
        "model": "hrrr",
        "product": "sfc",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hrrr_ncep_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hrrr.gem",
        "model_ext": "_hrrr.gem",
        "fxx": range(0, 49, 1),
        "params": None,
    },
]


def round_down_to_00_or_12(dt: datetime) -> datetime:
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0 if dt.hour < 12 else 12)


def extract_forecast_hour(filename: str) -> int | None:
    match = re.search(r"f(\d{2,})", filename)
    if match:
        return int(match.group(1))
    return None


def build_urls(cfg: dict, date: datetime | None = None) -> list[str]:
    run_date = round_down_to_00_or_12(date or datetime.utcnow())
    return [cfg["nssl_link"].format(date=run_date, fxx=fxx) for fxx in cfg["fxx"]]


def download_file(url: str, cfg: dict) -> str | None:
    filename = os.path.basename(url)
    parts = filename.split("_")[-1]
    out_dir = cfg["gribdir"]
    file_prefix = out_dir.split("/")[-1]
    filename = f"{file_prefix}_{parts}".replace("f0", "f")
    dest_path = os.path.join(out_dir, filename)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        print(f"Downloaded: {dest_path}")
        return dest_path
    except Exception as exc:
        print(f"Failed: {url} ({exc})")
        return None


def process_model(model_cfg: dict, max_workers: int = 4) -> None:
    print(f"\n=== Downloading {model_cfg['model'].upper()} ===")
    urls = build_urls(model_cfg)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, model_cfg) for url in urls]
        for future in as_completed(futures):
            _ = future.result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model token to process, e.g. hrrr or wrf4nssl")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)

    matched = False
    for cfg in MODELS:
        if args.model in cfg["gempak_pattern"] or args.model == cfg["model"]:
            process_model(cfg, max_workers=args.max_workers)
            matched = True

    if not matched:
        raise SystemExit(f"No model configuration matched {args.model!r}")

    return 0


__all__ = [
    "MODELS",
    "build_urls",
    "download_file",
    "extract_forecast_hour",
    "main",
    "process_model",
    "round_down_to_00_or_12",
]
