"""Remote model source adapters."""

from EnsDataStore.sources.remote.common import RemoteZarrSource, open_remote_zarr
from EnsDataStore.sources.remote.ecens_dynamical import SOURCE as ECENS_DYNAMICAL_SOURCE
from EnsDataStore.sources.remote.gefs_dynamical import SOURCE as GEFS_DYNAMICAL_SOURCE
from EnsDataStore.sources.remote.gfs_dynamical import SOURCE as GFS_DYNAMICAL_SOURCE
from EnsDataStore.sources.remote.gefsnssl_ensemble import load_gefs_ensemble

__all__ = [
    "ECENS_DYNAMICAL_SOURCE",
    "GEFS_DYNAMICAL_SOURCE",
    "GFS_DYNAMICAL_SOURCE",
    "RemoteZarrSource",
    "load_gefs_ensemble",
    "open_remote_zarr",
]
