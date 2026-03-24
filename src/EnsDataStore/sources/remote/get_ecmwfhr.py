"""Download and convert ECMWF high-resolution forecast data."""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
from datetime import datetime, timedelta

def round_down_to_00_or_12(dt: datetime) -> datetime:
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0 if dt.hour < 12 else 12)


def extract_forecast_hour(filename: str) -> int | None:
    match = re.search(r"f(\d{2,})", filename)
    if match:
        return int(match.group(1))
    return None

def process_model(model_cfg: dict) -> None:
    from herbie import FastHerbie

    model = model_cfg["model"]
    product = model_cfg["product"]
    gribdir = model_cfg["gribdir"]

    print(f"\n=== Processing {model.upper()} {product} ===")
    latest_run = round_down_to_00_or_12(datetime.utcnow() - timedelta(hours=8))
    print(latest_run)

    grid_name = gribdir.split("/")[-1]
    new_filename = f"{gribdir}/{grid_name}_{latest_run:%Y%m%d%H}f00.grib2"
    if os.path.exists(new_filename):
        return

    downloader = FastHerbie(
        [latest_run],
        model=model,
        product=product,
        fxx=model_cfg["fxx"],
        save_dir=model_cfg["gribdir"],
        priority=["ecmwf", "aws", "azure"],
        verbose=True,
        max_threads=10,
    )

    # Download the ECMWF HIRES GRIB2 DATA
    if model_cfg["params"] is not None:
        downloader.download(model_cfg["params"], verbose=False)
    else:
        downloader.download(verbose=False)

    for herbie_object in downloader.objects:
        path_to_file = str(herbie_object.get_localFilePath())
        print(f"Processing file: {path_to_file}")


def main(argv: list[str] | None = None) -> int:
    _ = argparse.ArgumentParser().parse_args(argv)
    for cfg in MODELS:
        process_model(cfg)
    return 0


__all__ = [
    "MODELS",
    "extract_forecast_hour",
    "main",
    "process_model",
    "round_down_to_00_or_12",
]
