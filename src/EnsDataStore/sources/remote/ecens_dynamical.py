"""Latest ECMWF ensemble dynamical forecast Zarr source."""

from __future__ import annotations

from EnsDataStore.sources.remote.common import RemoteZarrSource, open_remote_zarr

url = "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr"
SOURCE = RemoteZarrSource(
    name="ecens_dynamical",
    url=url,
    description="Latest ECMWF IFS ensemble dynamical forecast Zarr store.",
)


def open_latest_dataset(*, chunks=None, **kwargs):
    return open_remote_zarr(SOURCE, chunks=chunks, **kwargs)


__all__ = ["SOURCE", "open_latest_dataset", "url"]
