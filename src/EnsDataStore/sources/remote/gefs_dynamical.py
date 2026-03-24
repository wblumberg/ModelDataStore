"""Latest GEFS dynamical forecast Zarr source."""

from __future__ import annotations

from EnsDataStore.sources.remote.common import RemoteZarrSource, open_remote_zarr

url = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"
SOURCE = RemoteZarrSource(
    name="gefs_dynamical",
    url=url,
    description="Latest GEFS dynamical forecast Zarr store.",
)


def open_latest_dataset(*, chunks=None, **kwargs):
    return open_remote_zarr(SOURCE, chunks=chunks, **kwargs)


__all__ = ["SOURCE", "open_latest_dataset", "url"]
