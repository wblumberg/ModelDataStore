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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import xarray as xr
import zarr

try:
    import pint
    import pint_xarray
    _PINT_AVAILABLE = True
    _UNIT_REGISTRY = pint.UnitRegistry()
    pint_xarray.setup_registry(_UNIT_REGISTRY)
except ImportError:
    _PINT_AVAILABLE = False
    _UNIT_REGISTRY = None

try:
    import dask.array as da
    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional scipy import – only required for neighbourhood probability.
# If scipy is not installed the NP products are skipped with a warning.
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    from scipy.ndimage import maximum_filter as _scipy_max_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

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
    da = source.mean(dim=MEMBER_DIM, skipna=True)
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

    member_rolling_max = (
        source
        .rolling({TIME_DIM: window_hours}, min_periods=1)
        .max()
    )
    # Ensemble max over members → (time, y, x)
    ens_max = member_rolling_max.max(dim=MEMBER_DIM, skipna=True)
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

    member_rolling_min = (
        source
        .rolling({TIME_DIM: window_hours}, min_periods=1)
        .min()
    )
    ens_min = member_rolling_min.min(dim=MEMBER_DIM, skipna=True)
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

def _build_disk_kernel(radius_gridpoints: int) -> np.ndarray:
    """Return a boolean disk structuring element with radius in grid points.

    The kernel is used as a footprint for scipy's maximum_filter so that a
    grid point is considered "exceeded" if *any* point within the disk
    exceeds the threshold.
    """
    size = 2 * radius_gridpoints + 1
    cy, cx = radius_gridpoints, radius_gridpoints
    y_idx, x_idx = np.ogrid[:size, :size]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    return (dist <= radius_gridpoints).astype(np.float64)


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
    if not _SCIPY_AVAILABLE:
        logger.warning(
            "scipy not available – skipping neighbourhood probability for %s > %g",
            var_name,
            threshold,
        )
        return None

    radius_gp = max(1, round(radius_km / grid_spacing_km))
    kernel = _build_disk_kernel(radius_gp)
    logger.debug(
        "NB kernel: radius=%.0f km / %.1f km/gp → %d gp, footprint=%dx%d",
        radius_km,
        grid_spacing_km,
        radius_gp,
        kernel.shape[0],
        kernel.shape[1],
    )

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

    # Binary exceedance mask; keep this lazy (dask-backed) and chunk-aware.
    exceedance = rolling_max > threshold if strict else rolling_max >= threshold
    exceedance = exceedance.transpose(MEMBER_DIM, TIME_DIM, Y_DIM, X_DIM).astype(np.float32)

    footprint_4d = kernel.astype(bool)[np.newaxis, np.newaxis, :, :]

    # Use dask.map_overlap for chunk-aware spatial neighbourhood filtering.
    # Depth is only in y/x so time/member chunking is preserved.
    if _DASK_AVAILABLE and hasattr(exceedance.data, "map_overlap"):
        axis_y = exceedance.get_axis_num(Y_DIM)
        axis_x = exceedance.get_axis_num(X_DIM)

        hit_data = da.map_overlap(
            lambda block: _scipy_max_filter(
                block,
                footprint=footprint_4d,
                mode="constant",
                cval=0.0,
            ),
            exceedance.data,
            depth={axis_y: radius_gp, axis_x: radius_gp},
            boundary={axis_y: 0.0, axis_x: 0.0},
            dtype=np.float32,
            trim=True,
        )
        hit_mask = xr.DataArray(
            hit_data,
            dims=exceedance.dims,
            coords=exceedance.coords,
            attrs=exceedance.attrs,
            name=exceedance.name,
        )
    else:
        # NumPy fallback path when dask isn't available.
        hit_np = _scipy_max_filter(
            exceedance.values,
            footprint=footprint_4d,
            mode="constant",
            cval=0.0,
        )
        hit_mask = xr.DataArray(
            hit_np,
            dims=exceedance.dims,
            coords=exceedance.coords,
            attrs=exceedance.attrs,
            name=exceedance.name,
        )

    # Fraction of members with a neighbourhood hit at each (time, y, x).
    prob_fraction = (hit_mask >= 1.0).mean(dim=MEMBER_DIM, skipna=True).astype(np.float32)

    # Convert to percentage and smooth with a 2-D Gaussian kernel.
    # Smoothing parameter is 40 km -> sigma in grid points.
    sigma_gp = float(radius_km / grid_spacing_km)
    prob_percent = prob_fraction * 100.0

    if _DASK_AVAILABLE and hasattr(prob_percent.data, "map_overlap"):
        axis_y = prob_percent.get_axis_num(Y_DIM)
        axis_x = prob_percent.get_axis_num(X_DIM)
        depth = int(np.ceil(3.0 * sigma_gp))
        smooth_data = da.map_overlap(
            lambda block: _scipy_gaussian_filter(block, sigma=(0.0, sigma_gp, sigma_gp), mode="nearest"),
            prob_percent.data,
            depth={axis_y: depth, axis_x: depth},
            boundary={axis_y: "nearest", axis_x: "nearest"},
            dtype=np.float32,
            trim=True,
        )
        out = xr.DataArray(
            smooth_data,
            dims=prob_percent.dims,
            coords=prob_percent.coords,
            attrs=prob_percent.attrs,
            name=prob_percent.name,
        )
    else:
        out = xr.DataArray(
            _scipy_gaussian_filter(prob_percent.values, sigma=(0.0, sigma_gp, sigma_gp), mode="nearest"),
            dims=prob_percent.dims,
            coords=prob_percent.coords,
            attrs=prob_percent.attrs,
            name=prob_percent.name,
        )

    out = out.clip(min=0.0, max=100.0).astype(np.float32)

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


def compute_href_stats(
    ensemble: xr.Dataset,
    run_id: str = "",
    grid_spacing_km: float = 3.0,
) -> xr.Dataset:
    """Run all registered products and return a statistics Dataset.

    Each product function is called in order.  Variables not present in the
    ensemble are skipped with a warning so that a partially-populated ensemble
    does not halt the entire run.

    Parameters
    ----------
    ensemble:
        Time-lagged ensemble Dataset with dims (time, member, y, x).
    run_id:
        Cycle identifier written into the output metadata (e.g. "2026032200").
    grid_spacing_km:
        Grid spacing in km used for neighbourhood radius calculations.
    """
    products = build_product_list(grid_spacing_km=grid_spacing_km)

    member_names = [str(m) for m in ensemble.coords[MEMBER_DIM].values]
    n_members = len(member_names)

    output_vars: dict[str, xr.DataArray] = {}

    for i, entry in enumerate(products, start=1):
        fn = entry["fn"]
        try:
            da = fn(ensemble)
        except KeyError as exc:
            # A required source variable is missing from the ensemble.
            logger.warning(
                "[%d/%d] Skipping product (variable not found in ensemble): %s",
                i, len(products), exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%d/%d] Error computing product: %s", i, len(products), exc,
                exc_info=True,
            )
            continue

        if da is None:
            # Neighbourhood probability skipped due to missing scipy.
            continue

        key = da.name or f"product_{i}"
        if key in output_vars:
            logger.warning("Duplicate output key '%s'; overwriting previous result.", key)
        output_vars[key] = da
        logger.info("[%d/%d] Computed  %s", i, len(products), key)

    # Assemble the output Dataset
    stats_ds = xr.Dataset(output_vars)

    # Preserve coordinate metadata (units, etc.) from the source ensemble
    for coord in (TIME_DIM, Y_DIM, X_DIM):
        if coord in ensemble.coords and coord in stats_ds.coords:
            stats_ds[coord].attrs.update(ensemble.coords[coord].attrs)

    # Write ensemble membership and grid metadata as dataset-level attributes
    grid_attrs = _extract_grid_attrs(ensemble)
    stats_ds.attrs.update(
        {
            "title": "HREF Time-Lagged Ensemble – Post-Processing Statistics",
            "run_id": run_id,
            "created_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_members": n_members,
            "members": ",".join(member_names),
            "grid_spacing_km": grid_spacing_km,
            **grid_attrs,
        }
    )
    return stats_ds


# ===========================================================================
# Step 5 – Write output Zarr store
# ===========================================================================

def write_href_stats(
    stats_ds: xr.Dataset,
    output_path: Path,
    *,
    time_chunk: int = 4,
    y_chunk: int = 384,
    x_chunk: int = 384,
) -> None:
    """Write the statistics Dataset to a consolidated Zarr v2 store.

    The parent directory is created if it does not already exist.  Any
    existing store at *output_path* is overwritten.
    """
    # Ensure all variables have Zarr-compatible uniform chunking.
    # This is required because some map_overlap paths can produce irregular
    # dask chunk boundaries that Zarr rejects.
    chunk_spec = {
        TIME_DIM: time_chunk,
        Y_DIM: y_chunk,
        X_DIM: x_chunk,
    }
    stats_ds = stats_ds.chunk(chunk_spec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Writing output Zarr store to %s with chunks (time=%d, y=%d, x=%d) …",
        output_path,
        time_chunk,
        y_chunk,
        x_chunk,
    )

    # Write one variable at a time. The first variable creates the store
    # (including coords + global attrs), then remaining variables are appended.
    # This avoids constructing one very large dask graph for all variables at
    # once and gives visible progress during long writes.
    # Keys that frequently collide between attrs and encoding during CF/Zarr
    # serialization. We strip them from attrs and encoding before writing.
    encoding_attr_keys = {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "dtype",
    }

    def _sanitize_dataset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
        """Return a dataset safe for xarray -> zarr serialization."""
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

    data_var_names = list(stats_ds.data_vars)
    if not data_var_names:
        raise ValueError("No data variables to write")

    for idx, var_name in enumerate(data_var_names, start=1):
        var_ds = xr.Dataset({var_name: stats_ds[var_name]})
        # Keep root attrs stable across append writes. Some xarray/zarr paths
        # can replace store attrs with attrs from the incoming dataset.
        var_ds.attrs.update(stats_ds.attrs)
        var_ds = _sanitize_dataset_for_zarr(var_ds)
        var_encoding = _build_zarr_encoding(var_ds, include_coords=(idx == 1))
        logger.info("Writing variable %d/%d: %s", idx, len(stats_ds.data_vars), var_name)
        var_ds.to_zarr(
            str(output_path),
            mode="w" if idx == 1 else "a",
            consolidated=False,
            zarr_format=2,
            encoding=var_encoding,
        )

    # Consolidate once at the end (much cheaper than doing it for every append).
    zarr.consolidate_metadata(str(output_path))
    logger.info("Successfully wrote %d variables to %s", len(stats_ds.data_vars), output_path)


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
        default=1,
        help="Output write chunk size for the time dimension",
    )
    parser.add_argument(
        "--y-chunk",
        type=int,
        default=384,
        help="Output write chunk size for the y dimension",
    )
    parser.add_argument(
        "--x-chunk",
        type=int,
        default=384,
        help="Output write chunk size for the x dimension",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    current_dir = Path(args.current)
    lagged_dir = Path(args.lagged)
    output_path = Path(args.output)

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
    # Step 2: Compute post-processing statistics
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Computing post-processing statistics ===")
    stats = compute_href_stats(
        ensemble,
        run_id=args.run_id,
        grid_spacing_km=args.grid_spacing_km,
    )
    logger.info(
        "Computed %d statistical products over %d members",
        len(stats.data_vars),
        stats.attrs["n_members"],
    )

    # ------------------------------------------------------------------
    # Step 3: Write the output Zarr store
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Writing output Zarr store ===")
    write_href_stats(
        stats,
        output_path,
        time_chunk=args.time_chunk,
        y_chunk=args.y_chunk,
        x_chunk=args.x_chunk,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
