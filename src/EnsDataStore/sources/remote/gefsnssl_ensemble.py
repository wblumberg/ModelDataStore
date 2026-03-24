"""Load GEFS post-processed ensemble forecasts from NSSL THREDDS."""

from __future__ import annotations

import datetime
import warnings

import requests
import xarray as xr

BASE_URL = (
    "https://data.nssl.noaa.gov/thredds/dodsC/FRDD/GEFS-v12/"
    "GEFS_v12_forecasts/GEFS_op"
)
CATALOG_BASE_URL = (
    "https://data.nssl.noaa.gov/thredds/catalog/FRDD/GEFS-v12/"
    "GEFS_v12_forecasts/GEFS_op"
)

CONTROL_MEMBER = ["c00"]
PERTURBATION_MEMBERS = [f"p{i:02d}" for i in range(1, 31)]
ALL_MEMBERS = CONTROL_MEMBER + PERTURBATION_MEMBERS
RUN_HOURS = ["12", "00"]


def get_latest_run(lookback_days: int = 2) -> tuple[str, str]:
    today = datetime.datetime.utcnow().date()
    for delta in range(lookback_days + 1):
        check_date = today - datetime.timedelta(days=delta)
        date_str = check_date.strftime("%Y%m%d")
        for cycle in RUN_HOURS:
            run_id = f"{date_str}_{cycle}Z"
            catalog_url = f"{CATALOG_BASE_URL}/{run_id}/catalog.html"
            try:
                response = requests.head(catalog_url, timeout=10)
                if response.status_code == 200:
                    print(f"Latest available run: {run_id}")
                    return date_str, cycle
            except requests.RequestException:
                continue

    raise RuntimeError(f"Could not find a valid GEFS run in the last {lookback_days} days.")


def build_member_url(date_str: str, cycle: str, member: str) -> str:
    run_id = f"{date_str}_{cycle}Z"
    init_str = f"{date_str}{cycle}"
    filename = f"convective_parms_{init_str}_{member}_f000-f384.nc"
    return f"{BASE_URL}/{run_id}/{member}/{filename}"


def open_member_dataset(url: str, member: str) -> xr.Dataset | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dataset = xr.open_dataset(url, engine="pydap", chunks={})
            return dataset.expand_dims(dim="member").assign_coords(member=[member])
        except Exception as exc:
            print(f"  WARNING: could not open {member} - {exc}")
            return None


def load_gefs_ensemble(
    date_str: str | None = None,
    cycle: str | None = None,
    members: list[str] | None = None,
) -> xr.Dataset:
    if date_str is None or cycle is None:
        date_str, cycle = get_latest_run()

    if members is None:
        members = ALL_MEMBERS

    datasets: list[xr.Dataset] = []
    print(f"Opening {len(members)} member(s) for {date_str}_{cycle}Z ...")
    for member in members:
        url = build_member_url(date_str, cycle, member)
        print(f"  {member}: {url}")
        dataset = open_member_dataset(url, member)
        if dataset is not None:
            datasets.append(dataset)

    if not datasets:
        raise RuntimeError("No member datasets could be opened.")

    print(f"\nConcatenating {len(datasets)} member(s) along 'member' dimension ...")
    ensemble = xr.concat(datasets, dim="member")
    print("Done!  Ensemble dataset:")
    return ensemble


def main() -> int:
    ensemble = load_gefs_ensemble()
    ensemble_mean = ensemble.mean(dim="member")
    print("\nEnsemble mean t2m (first time step, degrees K):")
    print(ensemble_mean["t2m"].isel(time=0).values)
    return 0


__all__ = [
    "ALL_MEMBERS",
    "BASE_URL",
    "CATALOG_BASE_URL",
    "CONTROL_MEMBER",
    "PERTURBATION_MEMBERS",
    "RUN_HOURS",
    "build_member_url",
    "get_latest_run",
    "load_gefs_ensemble",
    "main",
    "open_member_dataset",
]
