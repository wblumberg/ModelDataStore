"""Common helpers for remote model sources."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class RemoteZarrSource:
    name: str
    url: str
    description: str = ""


def open_remote_zarr(
    source: RemoteZarrSource | str,
    *,
    chunks=None,
    **kwargs,
) -> xr.Dataset:
    url = source.url if isinstance(source, RemoteZarrSource) else source
    return xr.open_zarr(url, chunks=chunks, **kwargs)
