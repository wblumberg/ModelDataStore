#!/usr/bin/env python3
"""Build a Zarr store of HREF ensemble post-processing statistics.

This script:
  1. Opens member-level Zarr stores from two run-cycle directories (current
     + lagged) and concatenates them along the ``member`` dimension to form
     a time-lagged ensemble dataset.
  2. Computes a suite of deterministic ensemble statistics defined in the
     PRODUCTS list at the bottom of the file:
       - Ensemble means (temperature, dewpoint, winds, CAPE/CIN, SRH, MSLP,
         1-hr QPF)
       - Rolling 4-hour ensemble extremes (UH, updraft, downdraft, wind)
       - Neighbourhood exceedance probabilities (40-km radius, 4-hr window)
  3. Writes all statistics to a single output Zarr store that also carries
     the source grid metadata and ensemble membership information as
     dataset-level attributes.

Adding new post-processing products
-------------------------------------
Write a function with signature::

    my_product(ds: xr.Dataset) -> xr.DataArray

where *ds* is the full (time, member, y, x) ensemble Dataset and the
returned DataArray has dims ``(time, y, x)``.  Set ``da.name`` and
``da.attrs`` inside the function, then add an entry to the PRODUCTS list
near the bottom of the file.

Usage example::

    python build_href.py \\
        --current  data/zarr/2026032200.href_members \\
        --lagged   data/zarr/2026032112.href_members \\
        --output   data/zarr/2026032200.href_stats \\
        --run-id   2026032200
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from EnsDataStore.calc import ensemble as ensemble_calc

try:
    import dask
    _DASK_AVAILABLE = True
except ImportError:
    dask = None
    _DASK_AVAILABLE = False

try:
    from dask.distributed import Client, LocalCluster
    _DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    Client = None
    LocalCluster = None
    _DASK_DISTRIBUTED_AVAILABLE = False

try:
    import pint
    import pint_xarray
    _PINT_AVAILABLE = True
    _UNIT_REGISTRY = pint.UnitRegistry()
    pint_xarray.setup_registry(_UNIT_REGISTRY)
except ImportError:
    _PINT_AVAILABLE = False
    _UNIT_REGISTRY = None

logger = logging.getLogger(__name__)

# Dimension name constants – update here if your stores use different names.
MEMBER_DIM = "member"
TIME_DIM = "time"
Y_DIM = "y"
X_DIM = "x"

GRID_ATTR_KEYS: tuple[str, ...] = (
    "grid_type", "ni", "nj",
    "lon_0", "lat_0", "lat_std",
    "ll_lat", "ll_lon", "ur_lat", "ur_lon",
    "description", "source", "forecast_times", "member_name",
)

_LONGITUDE_ATTR_KEYS = {"lon_0", "ll_lon", "ur_lon"}


def _normalize_unit_string(raw_unit: str) -> str:
    """Normalize common GRIB-style unit strings into pint-friendly units."""
    text = raw_unit.strip()
    table = {
        "K": "kelvin",
        "m s-1": "meter / second",
        "m s^-1": "meter / second",
        "m/s": "meter / second",
        "m2 s-2": "meter**2 / second**2",
        "m^2 s^-2": "meter**2 / second**2",
        "Pa": "pascal",
    }
    return table.get(text, text)


def _convert_units_with_pint(
    da: xr.DataArray,
    target_unit: str,
    *,
    display_unit: str | None = None,
    source_unit_override: str | None = None,
) -> xr.DataArray:
    """Convert DataArray units using pint-xarray quantify/to/dequantify."""
    source_raw = source_unit_override or str(da.attrs.get("units", "")).strip()
    if not source_raw:
        raise ValueError(f"Variable '{da.name}' has no units attribute; cannot convert with pint")

    source_unit = _normalize_unit_string(source_raw)
    target_pint = _normalize_unit_string(target_unit)

    if not _PINT_AVAILABLE:
        raise RuntimeError(
            "pint and pint-xarray are required for unit-aware conversions. "
            "Install with: pip install pint pint-xarray"
        )

    # pint-xarray performs unit conversion on lazy/dask-backed arrays too.
    # Important: remove the original GRIB-style units attr before quantify,
    # otherwise pint-xarray may still parse the raw attr (e.g. "m s-1") and
    # fail even when a normalized unit is passed explicitly.
    da_for_quant = da.copy(deep=False)
    da_for_quant.attrs = dict(da.attrs)
    da_for_quant.attrs.pop("units", None)

    quantified: xr.DataArray | None = None

    # pint-xarray API varies slightly across versions; try common signatures.
    quantify_errors: list[Exception] = []
    for attempt in (
        lambda arr: arr.pint.quantify(source_unit, unit_registry=_UNIT_REGISTRY),
        lambda arr: arr.pint.quantify(units=source_unit, unit_registry=_UNIT_REGISTRY),
    ):
        try:
            quantified = attempt(da_for_quant)
            break
        except Exception as exc:  # noqa: BLE001
            quantify_errors.append(exc)

    if quantified is None:
        raise RuntimeError(
            f"Failed to quantify variable '{da.name}' with units '{source_unit}'. "
            f"Errors: {[str(e) for e in quantify_errors]}"
        )

    # In some versions quantify can return a DataArray without attached units
    # if the call signature is incompatible; guard this explicitly.
    if getattr(quantified.pint, "units", None) is None:
        # Last-resort fallback: set normalized units attr then quantify via attrs.
        temp = da_for_quant.copy(deep=False)
        temp.attrs["units"] = source_unit
        quantified = temp.pint.quantify(unit_registry=_UNIT_REGISTRY)

    if getattr(quantified.pint, "units", None) is None:
        raise RuntimeError(
            f"Variable '{da.name}' is not quantity-aware after quantify; "
            f"source units='{source_unit}'"
        )

    converted = quantified.pint.to(target_pint)
    out = converted.pint.dequantify()
    out.attrs.update(dict(da.attrs))
    out.attrs["units"] = display_unit or target_unit
    out.attrs["converted_with"] = "pint-xarray"
    out.attrs["source_units"] = source_raw
    return out


def _to_fahrenheit_with_pint(da: xr.DataArray) -> xr.DataArray:
    return _convert_units_with_pint(da, "degF", display_unit="degF")


def _to_knots_with_pint(da: xr.DataArray) -> xr.DataArray:
    return _convert_units_with_pint(da, "knot", display_unit="kt")


def configure_dask_runtime(
    num_workers: int,
    threads_per_worker: int,
    use_distributed: bool = False,
) -> Client | None:
    """Configure local Dask execution for better Slurm CPU utilization.

    Returns an optional distributed ``Client`` that should be closed by the
    caller. If distributed is not available, falls back to the threaded
    scheduler with ``num_workers``.
    """
    if not _DASK_AVAILABLE:
        logger.info("dask not installed; using xarray default execution runtime")
        return None

    if num_workers <= 0:
        logger.info("Using default dask runtime settings")
        return None

    threads = max(1, threads_per_worker)

    if use_distributed and _DASK_DISTRIBUTED_AVAILABLE and Client is not None and LocalCluster is not None:
        try:
            # In HPC/Slurm environments with cgroups, dask.distributed's memory auto-detection
            # from cgroup limits often yields absurdly small values (e.g., 1 MiB) that conflict
            # with actual available RAM. Disable memory management entirely (memory_limit=None)
            # and turn off the monitor to avoid spurious worker pauses.
            cluster = LocalCluster(
                n_workers=num_workers,
                threads_per_worker=threads,
                processes=False,
                dashboard_address=None,
                memory_limit=None,  # Disable dask memory management in HPC/cgroup environments
            )
            # Disable worker memory monitoring to prevent spurious pauses from cgroup misconfiguration
            with dask.config.set({"distributed.worker.memory.monitor": False}):
                client = Client(cluster)
            
            logger.info(
                "Configured local dask.distributed runtime: workers=%d, threads/worker=%d, total_threads=%d (memory management disabled for HPC)",
                num_workers,
                threads,
                num_workers * threads,
            )
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unable to start dask.distributed local cluster; falling back to threaded scheduler: %s",
                exc,
            )
    elif use_distributed:
        logger.warning(
            "--use-dask-distributed requested but dask.distributed is unavailable; using threaded scheduler",
        )

    # Fallback for environments without dask.distributed.
    dask.config.set(scheduler="threads", num_workers=num_workers * threads)
    logger.info(
        "Configured dask threaded scheduler: num_workers=%d",
        num_workers * threads,
    )
    return None


# ===========================================================================
# Step 1 – Load and assemble the time-lagged ensemble
# ===========================================================================

def discover_member_stores(root: Path, pattern: str = "*.zarr") -> list[Path]:
    """Return a sorted list of member Zarr store paths found under *root*."""
    stores = sorted(root.glob(pattern))
    if not stores:
        logger.warning("No stores matching '%s' found under %s", pattern, root)
    return stores


def _coerce_attr_value(value: object) -> object:
    """Convert NumPy scalar attrs to plain Python values for stable metadata."""
    return value.item() if hasattr(value, "item") else value


def _normalize_longitude_degrees(value: object) -> object:
    """Normalize numeric longitude values to [-180, 180) if possible."""
    try:
        return ((float(value) + 180.0) % 360.0) - 180.0
    except (TypeError, ValueError):
        return value


def _collect_grid_attrs_from_members(
    datasets: list[xr.Dataset],
    store_paths: list[Path],
) -> dict[str, object]:
    """Collect grid attrs from member stores with conflict warnings."""
    grid_attrs: dict[str, object] = {}

    for key in GRID_ATTR_KEYS:
        observed: list[tuple[Path, object]] = []
        for ds, path in zip(datasets, store_paths):
            if key not in ds.attrs:
                continue
            value = _coerce_attr_value(ds.attrs[key])
            if key in _LONGITUDE_ATTR_KEYS:
                value = _normalize_longitude_degrees(value)
            if value in ("", None):
                continue
            observed.append((path, value))

        if not observed:
            continue

        selected = observed[0][1]
        grid_attrs[key] = selected

        distinct = {repr(v) for _, v in observed}
        if len(distinct) > 1:
            logger.warning(
                "Conflicting grid attr '%s' across member stores; using %r from %s",
                key,
                selected,
                observed[0][0].name,
            )
            

    return grid_attrs


def _tag_member_names(
    datasets: list[xr.Dataset],
    current_stores: list[Path],
    lagged_stores: list[Path],
) -> list[xr.Dataset]:
    """Assign unique member coordinate values across current and lagged cycles.

    When the same model (e.g. ``hrrr``) appears in both the current and the
    lagged cycle directory its member coordinate value is made unique by
    appending a ``_cur`` or ``_lag`` suffix.  If all member names are already
    unique across both cycles, the original coordinates are left untouched.
    """
    n_current = len(current_stores)
    raw_names: list[str] = []
    for ds in datasets:
        if MEMBER_DIM in ds.coords:
            raw_names.append(str(ds.coords[MEMBER_DIM].values[0]))
        else:
            raw_names.append(ds.attrs.get("member_name", "unknown"))

    # Check whether any name is duplicated across the combined list
    if len(set(raw_names)) == len(raw_names):
        # All unique – no tagging required
        return datasets

    # Tag each member with its cycle origin to guarantee uniqueness
    tagged: list[xr.Dataset] = []
    for idx, (ds, raw) in enumerate(zip(datasets, raw_names)):
        suffix = "_cur" if idx < n_current else "_lag"
        new_name = f"{raw}{suffix}"
        tagged.append(ds.assign_coords({MEMBER_DIM: [new_name]}))
        logger.debug("  Tagged member %s → %s", raw, new_name)
    return tagged




def build_time_lagged_ensemble(
    current_dir: Path,
    lagged_dir: Path,
    *,
    member_pattern: str = "*.zarr",
    concat_join: str = "inner",
) -> xr.Dataset:
    """Open all member stores and concatenate into a single ensemble Dataset.

    Members from ``current_dir`` and ``lagged_dir`` are combined so that both
    the current-cycle and time-lagged members participate in every statistic.
    Each individual Zarr store has ``member`` size 1; after concatenation the
    dataset has dims ``(time, N_members, y, x)``.

    Parameters
    ----------
    current_dir:
        Directory containing current-cycle member Zarr stores.
    lagged_dir:
        Directory containing lagged-cycle member Zarr stores.
    member_pattern:
        Glob pattern used to discover individual member Zarr stores.
    concat_join:
        Xarray join strategy when aligning time coordinates across members.
        ``"inner"`` retains only times shared by *all* members (safe default).
        Use ``"outer"`` to keep all times and allow NaNs for missing members.
    """
    current_stores = discover_member_stores(current_dir, member_pattern)
    lagged_stores = discover_member_stores(lagged_dir, member_pattern)
    all_store_paths = current_stores + lagged_stores

    if not all_store_paths:
        raise FileNotFoundError(
            f"No member stores found under {current_dir} or {lagged_dir}"
        )

    logger.info(
        "Discovered %d current-cycle + %d lagged-cycle member stores",
        len(current_stores),
        len(lagged_stores),
    )

    # Open each store lazily (chunks="auto" enables dask-backed arrays).
    datasets: list[xr.Dataset] = []
    for path in all_store_paths:
        ds = xr.open_zarr(str(path), consolidated=True, chunks="auto")
        logger.debug(
            "  Opened %s  (member=%s, times=%d)",
            path.name,
            ds.coords[MEMBER_DIM].values[0] if MEMBER_DIM in ds.coords else "?",
            ds.sizes.get(TIME_DIM, 0),
        )
        datasets.append(ds)

    source_grid_attrs = _collect_grid_attrs_from_members(datasets, all_store_paths)

    # Concatenate along the pre-existing member dimension.  Each store has
    # member size 1, so this stacks them into (time, N_members, y, x).
    ensemble = xr.concat(
        _tag_member_names(datasets, current_stores, lagged_stores),
        dim=MEMBER_DIM,
        join=concat_join,
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

    if source_grid_attrs:
        # Ensure grid metadata in the concatenated ensemble reflects source
        # member stores rather than xarray concat attr resolution behavior.
        ensemble.attrs.update(source_grid_attrs)

    if TIME_DIM in ensemble.coords:
        ensemble = ensemble.sortby(TIME_DIM)

    logger.info(
        "Time-lagged ensemble assembled: %d members, %d times, grid %s×%s",
        ensemble.sizes[MEMBER_DIM],
        ensemble.sizes.get(TIME_DIM, 0),
        ensemble.sizes.get(X_DIM, "?"),
        ensemble.sizes.get(Y_DIM, "?"),
    )
    
    # This dataset is the source for all subsequent post-processing steps, so we return it from this function and pass it to the product functions defined below.  The products are responsible for selecting and transforming the relevant variables from the full ensemble dataset as needed.
    return ensemble


# ===========================================================================
# Step 2 – Individual post-processing functions
#
# Convention: every function accepts the full ensemble Dataset and returns
# an xr.DataArray with dims (time, y, x).  The function is responsible for
# setting ``da.name`` and ``da.attrs`` so the output Zarr is self-describing.
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Ensemble mean
# ---------------------------------------------------------------------------

def ensemble_mean(
    ds: xr.Dataset,
    var_name: str,
    long_name: str,
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    """Compute the ensemble mean for *var_name*, reducing over members.

    Parameters
    ----------
    ds:
        Full ensemble Dataset with dims (time, member, y, x).
    var_name:
        Name of the variable in *ds* to average.
    long_name:
        Human-readable description used in the output ``long_name`` attribute.
    """
    source = ds[var_name]
    if transform is not None:
        source = transform(source)
    da = ensemble_calc.mean_field(source, member_dim=MEMBER_DIM)
    da.name = f"mean_{var_name}"
    da.attrs.update(
        {
            "long_name": f"Ensemble mean {long_name}",
            "units": source.attrs.get("units", ""),
            "source_variable": var_name,
            "stat": "ensemble_mean",
        }
    )
    return da


# ---------------------------------------------------------------------------
# 2b. Rolling N-hour ensemble extremes
# ---------------------------------------------------------------------------

def rolling_window_ens_max(
    ds: xr.Dataset,
    var_name: str,
    long_name: str,
    window_hours: int = 4,
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    """Rolling *window_hours*-hour ensemble maximum at each grid point.

    For each time step the maximum is taken over:
      1. The preceding *window_hours* hourly values along the time dimension
         for each individual member (temporal rolling max).
      2. All ensemble members (ensemble max).

    This converts the stored 1-hour-maximum fields (e.g.
    ``MXUPHL_hght_5000_2000_max_1h``) into a *window_hours*-hour running
    maximum of the ensemble envelope.
    """
    # Temporal rolling max per member → (time, member, y, x)
    source = ds[var_name]
    if transform is not None:
        source = transform(source)

    ens_max = ensemble_calc.rolling_ensemble_max(
        source,
        window=window_hours,
        member_dim=MEMBER_DIM,
        time_dim=TIME_DIM,
        min_periods=1,
    )
    ens_max.name = f"max{window_hours}h_{var_name}"
    ens_max.attrs.update(
        {
            "long_name": f"{window_hours}-hr ensemble maximum {long_name}",
            "units": source.attrs.get("units", ""),
            "source_variable": var_name,
            "window_hours": window_hours,
            "stat": f"rolling_{window_hours}h_ens_max",
        }
    )
    return ens_max


def rolling_window_ens_min(
    ds: xr.Dataset,
    var_name: str,
    long_name: str,
    window_hours: int = 4,
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    """Rolling *window_hours*-hour ensemble minimum at each grid point.

    Mirrors :func:`rolling_window_ens_max` for minimum-valued fields such as
    the anti-cyclonic updraft helicity (``MNUPHL``) and downward vertical
    velocity (``MAXDVV``).
    """
    source = ds[var_name]
    if transform is not None:
        source = transform(source)

    ens_min = ensemble_calc.rolling_ensemble_min(
        source,
        window=window_hours,
        member_dim=MEMBER_DIM,
        time_dim=TIME_DIM,
        min_periods=1,
    )
    ens_min.name = f"min{window_hours}h_{var_name}"
    ens_min.attrs.update(
        {
            "long_name": f"{window_hours}-hr ensemble minimum {long_name}",
            "units": source.attrs.get("units", ""),
            "source_variable": var_name,
            "window_hours": window_hours,
            "stat": f"rolling_{window_hours}h_ens_min",
        }
    )
    return ens_min


# ---------------------------------------------------------------------------
# 2c. Neighbourhood exceedance probability
# ---------------------------------------------------------------------------

def neighbourhood_probability(
    ds: xr.Dataset,
    var_name: str,
    threshold: float,
    long_name: str,
    *,
    radius_km: float = 40.0,
    grid_spacing_km: float = 3.0,
    window_hours: int = 4,
    strict: bool = True,
    rolling_cache: dict[tuple[str, int], xr.DataArray] | None = None,
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray | None:
    """Neighbourhood exceedance probability within a rolling time window.

    Algorithm
    ---------
    For each valid time *t*:

    1. Per member: compute the rolling *window_hours*-hour maximum of
       *var_name* up to and including *t*.
    2. Per member: threshold the rolling max → binary exceedance mask.
    3. Per member: apply a spatial maximum filter with a disk footprint of
       radius ``radius_km / grid_spacing_km`` grid points.  A grid point
       gets value 1 if *any* point within the disk exceeded the threshold
       (neighbourhood "hit").
    4. Average the neighbourhood hit masks over all members → probability
       in [0, 1].

    Requires SciPy.  If SciPy is not installed the function logs a warning
    and returns ``None``; the calling loop will skip the product.

    Parameters
    ----------
    ds:
        Full ensemble Dataset with dims (time, member, y, x).
    var_name:
        Variable to threshold (must exist in *ds*).
    threshold:
        Exceedance threshold value (same units as the variable).
    long_name:
        Human-readable description of the variable.
    radius_km:
        Neighbourhood radius in kilometres (default 40 km for HREF).
    grid_spacing_km:
        Model grid spacing in kilometres (default 3 km for HREF).
    window_hours:
        Rolling accumulation window length in hours (default 4).
    strict:
        If ``True`` use strict greater-than (>); otherwise use ≥.
    """
    if not ensemble_calc.SCIPY_AVAILABLE:
        logger.warning(
            "scipy not available – skipping neighbourhood probability for %s > %g",
            var_name,
            threshold,
        )
        return None

    # Rolling max over time window → shape (time, member, y, x)
    # Cache this when running multiple thresholds for the same variable.
    cache_key = (var_name, window_hours, "transformed" if transform is not None else "raw")
    if rolling_cache is not None and cache_key in rolling_cache:
        rolling_max = rolling_cache[cache_key]
    else:
        source = ds[var_name]
        if transform is not None:
            source = transform(source)
        rolling_max = (
            source
            .rolling({TIME_DIM: window_hours}, min_periods=1)
            .max()
        )
        if rolling_cache is not None:
            rolling_cache[cache_key] = rolling_max

    out = ensemble_calc.neighborhood_probability_smoothed(
        rolling_max,
        threshold=threshold,
        member_dim=MEMBER_DIM,
        y_dim=Y_DIM,
        x_dim=X_DIM,
        radius_km=radius_km,
        grid_spacing_km=grid_spacing_km,
        strict=strict,
        smooth_sigma_km=radius_km,
        percentage=True,
    )

    thresh_str = f"{int(threshold)}" if threshold == int(threshold) else f"{threshold}"
    out.name = (
        f"prob_{var_name}_gt{thresh_str}_{window_hours}h_{int(radius_km)}km"
    )
    out.attrs.update(
        {
            "long_name": (
                f"Smoothed neighbourhood probability (%): {long_name} > {threshold} "
                f"({window_hours}-hr window, {int(radius_km)}-km radius, Gaussian sigma={int(radius_km)} km)"
            ),
            "units": "%",
            "source_variable": var_name,
            "threshold": float(threshold),
            "window_hours": window_hours,
            "neighbourhood_radius_km": float(radius_km),
            "grid_spacing_km": float(grid_spacing_km),
            "smoothing": "gaussian_2d",
            "smoothing_sigma_km": float(radius_km),
            "stat": "neighbourhood_probability",
        }
    )
    return out

# ===========================================================================
# Step 3 – Product registry
#
# Each entry is a dict with a "fn" key whose value is a callable
# ``(xr.Dataset) -> xr.DataArray | None``.
#
# To add a new product:
#   1. Write a post-processing function (or use the helpers above).
#   2. Add an entry below.  The DataArray's ``.name`` becomes the output
#      variable name in the Zarr store.
# ===========================================================================

def build_product_list(
    grid_spacing_km: float = 3.0,
) -> list[dict[str, Callable]]:
    """Return the ordered list of HREF post-processing product descriptors.

    Closure helpers keep individual entries terse.  Each closure captures
    the variable name and optional parameters, and returns a callable that
    accepts the ensemble Dataset and produces a DataArray.
    """

    # Neighbourhood radius and time window used for all NB-prob products.
    NB_RADIUS_KM: float = 40.0
    NB_WINDOW_HOURS: int = 4
    rolling_cache: dict[tuple[str, int], xr.DataArray] = {}

    # Closure wrappers ---------------------------------------------------------

    def mean(
        var: str,
        long_name: str,
        transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> Callable:
        """Ensemble mean factory."""
        return lambda ds: ensemble_mean(ds, var, long_name, transform=transform)

    def max4h(
        var: str,
        long_name: str,
        transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> Callable:
        """4-hr rolling ensemble maximum factory."""
        return lambda ds: rolling_window_ens_max(ds, var, long_name, window_hours=4, transform=transform)

    def min4h(
        var: str,
        long_name: str,
        transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> Callable:
        """4-hr rolling ensemble minimum factory."""
        return lambda ds: rolling_window_ens_min(ds, var, long_name, window_hours=4, transform=transform)

    def prob(
        var: str,
        threshold: float,
        long_name: str,
        transform: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> Callable:
        """Neighbourhood probability factory (40-km radius, 4-hr window)."""
        return lambda ds: neighbourhood_probability(
            ds,
            var,
            threshold,
            long_name,
            radius_km=NB_RADIUS_KM,
            grid_spacing_km=grid_spacing_km,
            window_hours=NB_WINDOW_HOURS,
            rolling_cache=rolling_cache,
            transform=transform,
        )

    # Product list -------------------------------------------------------------
    # Each dict requires only the "fn" key.  Add more keys (e.g. "priority")
    # if you later need to filter or order products programmatically.
    products: list[dict[str, Callable]] = [

        # ------------------------------------------------------------------ #
        # Ensemble mean fields                                                #
        # ------------------------------------------------------------------ #
        {"fn": mean("TMP_hght_2",         "2-m Temperature", transform=_to_fahrenheit_with_pint)},
        {"fn": mean("DPT_hght_2",         "2-m Dewpoint Temperature", transform=_to_fahrenheit_with_pint)},
        {"fn": mean("UGRD_hght_10",       "10-m U-Component of Wind", transform=_to_knots_with_pint)},
        {"fn": mean("VGRD_hght_10",       "10-m V-Component of Wind", transform=_to_knots_with_pint)},
        {"fn": mean("CAPE",               "CAPE (surface-based)")},
        {"fn": mean("CIN",                "CIN (surface-based)")},
        {"fn": mean("HLCY_hght_1000_0",   "Storm Relative Helicity 0–1 km")},
        {"fn": mean("HLCY_hght_3000_0",   "Storm Relative Helicity 0–3 km")},
        {"fn": mean("MSLMA",              "MSLP (MAPS System Reduction)",
                    transform=lambda da: da / 100.0)},  # hPa → mb
        {"fn": mean("APCP_accum_1h",      "1-hr Total Precipitation")},

        # ------------------------------------------------------------------ #
        # 4-hour rolling ensemble extremes                                    #
        # ------------------------------------------------------------------ #

        # Updraft helicity (cyclonic + anti-cyclonic 2–5 km)
        {"fn": max4h("MXUPHL_hght_5000_2000_max_1h",
                     "Updraft Helicity 2–5 km")},
        {"fn": min4h("MNUPHL_hght_5000_2000_min_1h",
                     "Min (Anti-Cyclonic) Updraft Helicity 2–5 km")},

        # Vertical velocity extremes
        {"fn": max4h("MAXUVV_max_1h",    "Upward Vertical Velocity")},
        {"fn": min4h("MAXDVV_max_1h",    "Downward Vertical Velocity")},

        # Near-surface wind speed
        {"fn": max4h("WIND_hght_10_max_1h", "10-m Wind Speed", transform=_to_knots_with_pint)},

        # ------------------------------------------------------------------ #
        # Neighbourhood exceedance probabilities                              #
        # (4-hr rolling window, 40-km radius)                                 #
        # ------------------------------------------------------------------ #

        # Updraft helicity thresholds (m² s⁻²)
        {"fn": prob("MXUPHL_hght_5000_2000_max_1h",  75,  "UH 2–5 km")},
        {"fn": prob("MXUPHL_hght_5000_2000_max_1h", 150,  "UH 2–5 km")},

        # Upward vertical velocity thresholds (m s⁻¹)
        {"fn": prob("MAXUVV_max_1h",                  20,  "Upward Vert. Velocity")},
        {"fn": prob("MAXUVV_max_1h",                  30,  "Upward Vert. Velocity")},

        # Near-surface wind speed thresholds (m s⁻¹)
        {"fn": prob("WIND_hght_10_max_1h",            30,  "10-m Wind Speed", transform=_to_knots_with_pint)},
        {"fn": prob("WIND_hght_10_max_1h",            50,  "10-m Wind Speed", transform=_to_knots_with_pint)},

        # ------------------------------------------------------------------ #
        # Placeholder – add new products here                                 #
        # Example: mean 1-km reflectivity for precipitation-type composites  #
        # {"fn": mean("REFD_hght_1000", "1-km Reflectivity")},               #
        # ------------------------------------------------------------------ #
    ]

    return products


# ===========================================================================
# Step 4 – Compute all products and assemble output Dataset
# ===========================================================================

def _extract_grid_attrs(ensemble: xr.Dataset) -> dict:
    """Collect grid-description attributes from the ensemble store metadata."""
    return {k: ensemble.attrs.get(k, "") for k in GRID_ATTR_KEYS}


def _sanitize_dataset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset safe for xarray -> zarr serialization."""
    encoding_attr_keys = {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "dtype",
    }
    cleaned = ds.copy(deep=False).drop_encoding()

    def _strip(da_in: xr.DataArray) -> xr.DataArray:
        da_out = da_in.copy(deep=False)
        for key in list(da_out.attrs):
            if key in encoding_attr_keys:
                da_out.attrs.pop(key, None)
        for key in list(da_out.encoding):
            if key in encoding_attr_keys:
                da_out.encoding.pop(key, None)
        return da_out

    for name in list(cleaned.data_vars):
        cleaned[name] = _strip(cleaned[name])
    for name in list(cleaned.coords):
        cleaned = cleaned.assign_coords({name: _strip(cleaned.coords[name])})

    for key in list(cleaned.attrs):
        if key in encoding_attr_keys:
            cleaned.attrs.pop(key, None)

    return cleaned


def _build_zarr_encoding(
    ds: xr.Dataset,
    *,
    time_chunk: int = 4,
    y_chunk: int = 384,
    x_chunk: int = 384,
    include_coords: bool = True,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Build explicit chunk encoding for every variable/coord in *ds*.

    This avoids zarr v2 metadata errors when xarray would otherwise pass
    chunks=None for some arrays.
    """
    dim_chunk_map = {
        TIME_DIM: time_chunk,
        Y_DIM: y_chunk,
        X_DIM: x_chunk,
    }
    encoding: dict[str, dict[str, tuple[int, ...]]] = {}
    variable_names: list[str]
    if include_coords:
        variable_names = list(ds.variables)
    else:
        variable_names = list(ds.data_vars)

    for name in variable_names:
        var = ds[name]
        if not var.dims:
            continue
        chunks: list[int] = []
        for dim in var.dims:
            target = dim_chunk_map.get(dim, ds.sizes[dim])
            chunks.append(int(min(ds.sizes[dim], target)))
        encoding[name] = {"chunks": tuple(chunks)}
    return encoding


def _write_dataset_batch_to_zarr(
    batch_ds: xr.Dataset,
    output_path: Path,
    *,
    is_first_write: bool,
    dataset_attrs: dict,
    time_chunk: int = 4,
    y_chunk: int = 384,
    x_chunk: int = 384,
    float_dtype: str = "float16",
    keep_float32_vars: set[str] | None = None,
    compression_codec: str = "zstd",
    compression_level: int = 5,
) -> tuple[float, float, list[str]]:
    """Write a batch Dataset to Zarr, returning prep/write timings and names.

    Parameters
    ----------
    batch_ds:
        Batch Dataset containing one or more output variables.
    output_path:
        Path to Zarr store
    is_first_write:
        If True, creates store with mode='w'; otherwise appends with mode='a'
    dataset_attrs:
        Global attributes to attach to the store
    time_chunk, y_chunk, x_chunk:
        Chunk sizes for spatial/temporal dimensions
    float_dtype:
        Target dtype for floating-point variables (float16 or float32)
    keep_float32_vars:
        Set of variable names to keep at float32
    compression_codec, compression_level:
        Compression settings for Blosc
    """
    if keep_float32_vars is None:
        keep_float32_vars = set()

    batch_start = perf_counter()
    var_names = list(batch_ds.data_vars)

    # Probability outputs: convert to uint8 percent [0, 100] to reduce disk.
    for var_name in var_names:
        if var_name.startswith("prob_"):
            batch_ds[var_name] = (
                batch_ds[var_name]
                .clip(min=0.0, max=100.0)
                .round()
                .astype(np.uint8)
            )

    # Keep root attrs stable across appends
    batch_ds.attrs.update(dataset_attrs)
    batch_ds = _sanitize_dataset_for_zarr(batch_ds)
    var_encoding = _build_zarr_encoding(
        batch_ds,
        time_chunk=time_chunk,
        y_chunk=y_chunk,
        x_chunk=x_chunk,
        include_coords=is_first_write,
    )

    # Set compression and dtype per variable
    compressor = Blosc(
        cname=compression_codec,
        clevel=int(compression_level),
        shuffle=Blosc.BITSHUFFLE,
    )
    for var_name in var_names:
        if var_name not in var_encoding:
            continue
        var_encoding[var_name]["compressor"] = compressor
        if var_name.startswith("prob_"):
            var_encoding[var_name]["dtype"] = "uint8"
        elif np.issubdtype(batch_ds[var_name].dtype, np.floating):
            target_dtype = "float32" if var_name in keep_float32_vars else float_dtype
            var_encoding[var_name]["dtype"] = target_dtype

    prep_done = perf_counter()
    prep_time = prep_done - batch_start

    # Write batch to Zarr (first write creates store, subsequent appends).
    batch_ds.to_zarr(
        str(output_path),
        mode="w" if is_first_write else "a",
        consolidated=False,
        zarr_format=2,
        encoding=var_encoding,
    )

    write_done = perf_counter()
    write_time = write_done - prep_done

    return prep_time, write_time, var_names


def process_and_write_products(
    ensemble: xr.Dataset,
    output_path: Path,
    *,
    run_id: str = "",
    grid_spacing_km: float = 3.0,
    time_chunk: int = 4,
    y_chunk: int = 384,
    x_chunk: int = 384,
    float_dtype: str = "float16",
    keep_float32_vars: Sequence[str] | None = None,
    compression_codec: str = "zstd",
    compression_level: int = 5,
    write_batch_size: int = 4,
) -> None:
    """Compute and write ensemble statistics to Zarr in bounded batches.

    This approach balances throughput and memory by computing/writing a small
    batch of products at a time, instead of strict one-by-one writes.

    Parameters
    ----------
    ensemble:
        Time-lagged ensemble Dataset with dims (time, member, y, x).
    output_path:
        Path to output Zarr store.
    run_id:
        Cycle identifier written into output metadata.
    grid_spacing_km:
        Grid spacing in km for neighbourhood calculations.
    time_chunk, y_chunk, x_chunk:
        Output chunk sizes for spatial/temporal dimensions.
    float_dtype:
        Target dtype for floating-point outputs.
    keep_float32_vars:
        Variable names to keep at float32 precision.
    compression_codec, compression_level:
        Blosc compression settings.
    write_batch_size:
        Number of products to compute/write together per batch.
    """
    products = build_product_list(grid_spacing_km=grid_spacing_km)
    if write_batch_size < 1:
        raise ValueError("write_batch_size must be >= 1")

    member_names = [str(m) for m in ensemble.coords[MEMBER_DIM].values]
    n_members = len(member_names)
    keep_float32 = {name for name in (keep_float32_vars or []) if name}

    # Prepare global dataset attributes (these go into every write)
    grid_attrs = _extract_grid_attrs(ensemble)
    dataset_attrs = {
        "title": "HREF Time-Lagged Ensemble – Post-Processing Statistics",
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_members": n_members,
        "members": ",".join(member_names),
        "grid_spacing_km": grid_spacing_km,
        **grid_attrs,
    }

    # Ensure stale store keys don't pollute the write
    if output_path.exists():
        logger.info("Removing existing output store before write: %s", output_path)
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Writing output Zarr store to %s with chunks (time=%d, y=%d, x=%d) …",
        output_path,
        time_chunk,
        y_chunk,
        x_chunk,
    )

    # Rolling cache persists across products to avoid recomputing rolling maxes
    # for related thresholds (e.g., multiple prob thresholds for same variable)
    rolling_cache: dict[tuple[str, int], xr.DataArray] = {}

    step3_start = perf_counter()
    products_written = 0

    chunk_spec = {
        TIME_DIM: time_chunk,
        Y_DIM: y_chunk,
        X_DIM: x_chunk,
    }

    for start in range(0, len(products), write_batch_size):
        end = min(start + write_batch_size, len(products))
        batch_entries = products[start:end]
        batch_vars: dict[str, xr.DataArray] = {}

        for i, entry in enumerate(batch_entries, start=start + 1):
            fn = entry["fn"]
            try:
                da = fn(ensemble)
            except KeyError as exc:
                logger.warning(
                    "[%d/%d] Skipping product (variable not found): %s",
                    i, len(products), exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%d/%d] Error computing product: %s",
                    i, len(products), exc,
                    exc_info=True,
                )
                continue

            if da is None:
                # Neighbourhood probability skipped due to missing scipy
                continue

            var_name = da.name or f"product_{i}"
            if var_name in batch_vars:
                logger.warning("Duplicate output key '%s'; overwriting batch value.", var_name)
            batch_vars[var_name] = da

        if not batch_vars:
            continue

        batch_ds = xr.Dataset(batch_vars).chunk(chunk_spec)
        for coord in (TIME_DIM, Y_DIM, X_DIM):
            if coord in ensemble.coords and coord in batch_ds.coords:
                batch_ds.coords[coord].attrs.update(ensemble.coords[coord].attrs)

        is_first = products_written == 0
        prep_time, write_time, written_names = _write_dataset_batch_to_zarr(
            batch_ds,
            output_path,
            is_first_write=is_first,
            dataset_attrs=dataset_attrs,
            time_chunk=time_chunk,
            y_chunk=y_chunk,
            x_chunk=x_chunk,
            float_dtype=float_dtype,
            keep_float32_vars=keep_float32,
            compression_codec=compression_codec,
            compression_level=compression_level,
        )

        products_written += len(written_names)
        logger.info(
            "Wrote batch %d-%d (%d vars); prep=%.2fs write=%.2fs",
            start + 1,
            end,
            len(written_names),
            prep_time,
            write_time,
        )

    # Consolidate metadata once at the end
    consolidate_start = perf_counter()
    zarr.consolidate_metadata(str(output_path))
    consolidate_done = perf_counter()
    step3_done = perf_counter()

    total_seconds = step3_done - step3_start
    mean_seconds = total_seconds / max(1, products_written)
    logger.info("Timing metadata consolidation: %.2fs", consolidate_done - consolidate_start)
    logger.info(
        "Timing Step 3 summary: total=%.2fs vars=%d avg_per_var=%.2fs",
        total_seconds,
        products_written,
        mean_seconds,
    )
    logger.info("Successfully wrote %d variables to %s", products_written, output_path)



# ===========================================================================
# Command-line interface
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Zarr store of HREF time-lagged ensemble post-processing "
            "statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--current",
        required=True,
        metavar="DIR",
        help="Directory containing current-cycle member .zarr stores",
    )
    parser.add_argument(
        "--lagged",
        required=True,
        metavar="DIR",
        help="Directory containing lagged-cycle member .zarr stores",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output Zarr store path (will be created or overwritten)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run cycle identifier embedded in output metadata (e.g. '2026032200')",
    )
    parser.add_argument(
        "--grid-spacing-km",
        type=float,
        default=3.0,
        help=(
            "Model grid spacing in km used for neighbourhood radius calculations "
            "(HREF CONUS grid ≈ 3 km)"
        ),
    )
    parser.add_argument(
        "--member-pattern",
        default="*.zarr",
        help="Glob pattern for discovering member Zarr stores",
    )
    parser.add_argument(
        "--concat-join",
        choices=("inner", "outer", "left", "right", "exact", "override"),
        default="inner",
        help="Xarray join strategy when aligning time coordinates across members",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=4,
        help="Output write chunk size for the time dimension",
    )
    parser.add_argument(
        "--y-chunk",
        type=int,
        default=353,
        help="Output write chunk size for the y dimension",
    ) # 353 x 3 = 1059, which is the full HREF CONUS grid size in y.  This ensures each chunk contains a full latitudinal row, which is ideal for map_overlap operations that need to access neighbouring rows.
    parser.add_argument(
        "--x-chunk",
        type=int,
        default=257,
        help="Output write chunk size for the x dimension",
    ) # 257 x 7 = 1799, which is the full HREF CONUS grid size in x.  This ensures each chunk contains a full longitudinal column, which is ideal for map_overlap operations that need to access neighbouring columns.
    parser.add_argument(
        "--float-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Target dtype for floating-point data variables at write time",
    )
    parser.add_argument(
        "--keep-float32-vars",
        default="",
        help="Comma-separated variable names to keep at float32 precision",
    )
    parser.add_argument(
        "--compression-codec",
        default="zstd",
        choices=("zstd", "lz4", "zlib", "blosclz"),
        help="Blosc codec used for output data variables",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Blosc compression level (0-9)",
    )
    parser.add_argument(
        "--dask-num-workers",
        type=int,
        default=0,
        help=(
            "Number of local Dask workers. Set >0 to configure runtime explicitly; "
            "0 uses default Dask settings"
        ),
    )
    parser.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=1,
        help="Threads per Dask worker when --dask-num-workers > 0",
    )
    parser.add_argument(
        "--use-dask-distributed",
        action="store_true",
        help=(
            "Use dask.distributed LocalCluster runtime. By default the script uses "
            "the threaded scheduler, which often performs better for large local graphs"
        ),
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=4,
        help=(
            "Number of products to compute/write per batch. "
            "Higher values are faster but use more memory"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "CPU visibility: os.cpu_count=%s, SLURM_CPUS_PER_TASK=%s",
        os.cpu_count(),
        os.environ.get("SLURM_CPUS_PER_TASK", "unset"),
    )

    current_dir = Path(args.current)
    lagged_dir = Path(args.lagged)
    output_path = Path(args.output)

    dask_client: Client | None = None
    dask_client = configure_dask_runtime(
        num_workers=args.dask_num_workers,
        threads_per_worker=args.dask_threads_per_worker,
        use_distributed=args.use_dask_distributed,
    )

    try:
        # ------------------------------------------------------------------
        # Step 1: Build the time-lagged ensemble from member Zarr stores
        # ------------------------------------------------------------------
        logger.info("=== Step 1: Loading time-lagged ensemble ===")
        ensemble = build_time_lagged_ensemble(
            current_dir=current_dir,
            lagged_dir=lagged_dir,
            member_pattern=args.member_pattern,
            concat_join=args.concat_join,
        )

        # ------------------------------------------------------------------
        # Step 2 & 3: Compute and write products (streaming, one at a time)
        # ------------------------------------------------------------------
        logger.info("=== Step 2 & 3: Computing and writing products ===")
        process_and_write_products(
            ensemble,
            output_path=output_path,
            run_id=args.run_id,
            grid_spacing_km=args.grid_spacing_km,
            time_chunk=args.time_chunk,
            y_chunk=args.y_chunk,
            x_chunk=args.x_chunk,
            float_dtype=args.float_dtype,
            keep_float32_vars=[v.strip() for v in args.keep_float32_vars.split(",") if v.strip()],
            compression_codec=args.compression_codec,
            compression_level=args.compression_level,
            write_batch_size=args.write_batch_size,
        )

        return 0
    finally:
        if dask_client is not None:
            dask_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
