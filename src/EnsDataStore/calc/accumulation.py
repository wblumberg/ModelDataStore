"""Utilities for converting run-accumulated fields for lagged-ensemble workflows."""

from __future__ import annotations

import numpy as np
import xarray as xr


def interval_accumulation(
    *,
    one_hour_accumulated: xr.DataArray | None = None,
    run_accumulated: xr.DataArray | None = None,
    time_dim: str = "time",
    source_cycle_var: xr.DataArray | None = None,
    source_forecast_hour_var: xr.DataArray | None = None,
    clip_negative: bool = True,
) -> xr.DataArray:
    """Return interval accumulation, preferring native 1-hour accumulation.

    If one_hour_accumulated is available, it is used directly.
    Otherwise this falls back to differencing run_accumulated across time with
    cycle-aware resets.
    """
    if one_hour_accumulated is not None:
        interval = one_hour_accumulated.copy(deep=False)
        if clip_negative:
            interval = xr.where(interval < 0, 0, interval)
        interval.attrs.update(dict(one_hour_accumulated.attrs))
        interval.attrs["accumulation_representation"] = "interval"
        interval.attrs["accumulation_source"] = "native_1h"
        interval.name = f"{one_hour_accumulated.name}_interval" if one_hour_accumulated.name else None
        return interval

    if run_accumulated is None:
        raise ValueError("Provide one_hour_accumulated or run_accumulated")

    return run_accumulated_to_interval(
        run_accumulated,
        time_dim=time_dim,
        source_cycle_var=source_cycle_var,
        source_forecast_hour_var=source_forecast_hour_var,
        clip_negative=clip_negative,
    )


def run_accumulated_to_interval(
    da: xr.DataArray,
    *,
    time_dim: str = "time",
    source_cycle_var: xr.DataArray | None = None,
    source_forecast_hour_var: xr.DataArray | None = None,
    clip_negative: bool = True,
) -> xr.DataArray:
    """Convert a run-accumulated field into interval increments.

    This uses first differences along time but resets at cycle boundaries.
    If source cycle is available, resets are detected from cycle changes.
    Otherwise, forecast hour decreases are used.
    """
    if time_dim not in da.dims:
        raise ValueError(f"time_dim '{time_dim}' not found in dims {da.dims}")

    diff = da.diff(time_dim)
    padded = diff.pad({time_dim: (1, 0)}, constant_values=np.nan)
    padded = padded.assign_coords({time_dim: da[time_dim]})

    reset_mask = _reset_mask(
        da=da,
        time_dim=time_dim,
        source_cycle_var=source_cycle_var,
        source_forecast_hour_var=source_forecast_hour_var,
    )

    interval = xr.where(reset_mask, da, padded)
    if clip_negative:
        interval = xr.where(interval < 0, 0, interval)

    interval.attrs.update(dict(da.attrs))
    interval.attrs["accumulation_representation"] = "interval"
    interval.attrs["accumulation_source"] = "run_accumulated"
    interval.name = f"{da.name}_interval" if da.name else None
    return interval


def interval_to_run_accumulated(
    interval: xr.DataArray,
    *,
    time_dim: str = "time",
    source_cycle_var: xr.DataArray | None = None,
    source_forecast_hour_var: xr.DataArray | None = None,
) -> xr.DataArray:
    """Reconstruct run-accumulated totals from interval increments.

    Accumulation is reset at cycle boundaries using source_cycle_var when
    available, otherwise source_forecast_hour_var decreases are treated as
    cycle resets.
    """
    if time_dim not in interval.dims:
        raise ValueError(f"time_dim '{time_dim}' not found in dims {interval.dims}")

    reset_mask = _reset_mask(
        da=interval,
        time_dim=time_dim,
        source_cycle_var=source_cycle_var,
        source_forecast_hour_var=source_forecast_hour_var,
    )
    group_id = reset_mask.astype("int32").cumsum(time_dim)
    run_total = interval.fillna(0).groupby(group_id).cumsum(time_dim)

    run_total.attrs.update(dict(interval.attrs))
    run_total.attrs["accumulation_representation"] = "run_total"
    run_total.attrs["accumulation_source"] = "interval"
    run_total.name = f"{interval.name}_run_total" if interval.name else None
    return run_total


def _reset_mask(
    *,
    da: xr.DataArray,
    time_dim: str,
    source_cycle_var: xr.DataArray | None,
    source_forecast_hour_var: xr.DataArray | None,
) -> xr.DataArray:
    first = xr.DataArray(
        np.zeros(da.sizes[time_dim], dtype=bool),
        dims=(time_dim,),
        coords={time_dim: da[time_dim]},
    )
    first[{time_dim: 0}] = True

    if source_cycle_var is not None:
        cycle_change = source_cycle_var != source_cycle_var.shift({time_dim: 1})
        return (cycle_change.fillna(True) | first).astype(bool)

    if source_forecast_hour_var is not None:
        fh = source_forecast_hour_var.astype("float64")
        cycle_change = fh < fh.shift({time_dim: 1})
        return (cycle_change.fillna(True) | first).astype(bool)

    return first
