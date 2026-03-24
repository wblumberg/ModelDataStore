"""Ensemble post-processing utilities for NWP workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr


DEFAULT_MEMBER_CANDIDATES: tuple[str, ...] = (
    "member",
    "ensemble",
    "ens",
    "realization",
    "number",
)

DEFAULT_TIME_CANDIDATES: tuple[str, ...] = (
    "time",
    "valid_time",
)

DEFAULT_WINDOW_DIM_CANDIDATES: tuple[str, ...] = (
    "time",
    "valid_time",
)

DEFAULT_Y_CANDIDATES: tuple[str, ...] = ("y", "lat", "latitude", "nj")
DEFAULT_X_CANDIDATES: tuple[str, ...] = ("x", "lon", "longitude", "ni")


def _find_dim(da: xr.DataArray, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in da.dims:
            return name
    raise ValueError(
        f"Could not identify required dimension from candidates {tuple(candidates)}. "
        f"Available dims: {da.dims}"
    )


def _find_coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    return None


def _normalize_longitudes(values: np.ndarray) -> np.ndarray:
    norm = ((values + 180.0) % 360.0) - 180.0
    return norm.astype(np.float64, copy=False)


def _iter_xy_dims(da: xr.DataArray, y_dim: str | None = None, x_dim: str | None = None) -> tuple[str, str]:
    if y_dim is not None and x_dim is not None:
        return y_dim, x_dim

    resolved_y = y_dim
    if resolved_y is None:
        for cand in DEFAULT_Y_CANDIDATES:
            if cand in da.dims:
                resolved_y = cand
                break

    resolved_x = x_dim
    if resolved_x is None:
        for cand in DEFAULT_X_CANDIDATES:
            if cand in da.dims:
                resolved_x = cand
                break

    if resolved_y is not None and resolved_x is not None:
        return resolved_y, resolved_x

    non_member_dims = [d for d in da.dims if d not in DEFAULT_MEMBER_CANDIDATES]
    if len(non_member_dims) < 2:
        raise ValueError(
            "Could not infer spatial dimensions; provide y_dim/x_dim explicitly. "
            f"Available dims: {da.dims}"
        )

    inferred_y = non_member_dims[-2]
    inferred_x = non_member_dims[-1]

    return resolved_y or inferred_y, resolved_x or inferred_x


def _ordered_quantile(da: xr.DataArray, q: float, dim: str) -> xr.DataArray:
    out = da.quantile(q, dim=dim, skipna=True)
    if "quantile" in out.dims:
        out = out.squeeze("quantile", drop=True)
    elif "quantile" in out.coords:
        out = out.drop_vars("quantile")
    return out


def _required_unsigned_dtype(bit_count: int) -> np.dtype:
    if bit_count <= 8:
        return np.dtype("uint8")
    if bit_count <= 16:
        return np.dtype("uint16")
    if bit_count <= 32:
        return np.dtype("uint32")
    if bit_count <= 64:
        return np.dtype("uint64")
    raise ValueError(
        "paintball_bitmask supports up to 64 members when packed into one integer field"
    )


def _resolve_paintball_dtype(
    member_count: int,
    output_dtype: str | np.dtype | None,
) -> np.dtype:
    required = _required_unsigned_dtype(member_count)
    if output_dtype is None:
        return required

    resolved = np.dtype(output_dtype)
    if resolved.kind != "u":
        raise ValueError("output_dtype must be an unsigned integer dtype")

    required_bits = member_count
    available_bits = resolved.itemsize * 8
    if available_bits < required_bits:
        raise ValueError(
            f"output_dtype {resolved} has {available_bits} bits, but {required_bits} are required"
        )
    return resolved


@dataclass(frozen=True)
class EnsembleDims:
    member_dim: str = "member"
    time_dim: str | None = "time"
    y_dim: str = "y"
    x_dim: str = "x"


def infer_ensemble_dims(
    da: xr.DataArray,
    member_dim: str | None = None,
    time_dim: str | None = None,
    y_dim: str | None = None,
    x_dim: str | None = None,
) -> EnsembleDims:
    resolved_member = member_dim or _find_dim(da, DEFAULT_MEMBER_CANDIDATES)

    if time_dim is not None:
        if time_dim not in da.dims:
            raise ValueError(f"time_dim '{time_dim}' not found in dims {da.dims}")
        resolved_time: str | None = time_dim
    else:
        resolved_time = next((d for d in DEFAULT_TIME_CANDIDATES if d in da.dims), None)

    resolved_y, resolved_x = _iter_xy_dims(da, y_dim=y_dim, x_dim=x_dim)
    return EnsembleDims(
        member_dim=resolved_member,
        time_dim=resolved_time,
        y_dim=resolved_y,
        x_dim=resolved_x,
    )


def mean_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
    return da.mean(dim=member_dim, skipna=True)


def spread_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
    return da.std(dim=member_dim, skipna=True)


def max_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
    return da.max(dim=member_dim, skipna=True)


def min_field(da: xr.DataArray, member_dim: str = "member") -> xr.DataArray:
    return da.min(dim=member_dim, skipna=True)


def probability_exceedance(
    da: xr.DataArray,
    threshold: float,
    member_dim: str = "member",
    strict: bool = True,
) -> xr.DataArray:
    mask = da > threshold if strict else da >= threshold
    return mask.mean(dim=member_dim, skipna=True)


def probability_in_range(
    da: xr.DataArray,
    lower: float,
    upper: float,
    member_dim: str = "member",
    include_lower: bool = True,
    include_upper: bool = False,
) -> xr.DataArray:
    left = da >= lower if include_lower else da > lower
    right = da <= upper if include_upper else da < upper
    return (left & right).mean(dim=member_dim, skipna=True)


def exceedance_fraction(
    da: xr.DataArray,
    threshold: float,
    member_dim: str = "member",
    strict: bool = True,
) -> xr.DataArray:
    return 100.0 * probability_exceedance(
        da=da,
        threshold=threshold,
        member_dim=member_dim,
        strict=strict,
    )


def decile_membership(
    da: xr.DataArray,
    decile: int,
    member_dim: str = "member",
) -> xr.DataArray:
    if decile < 1 or decile > 10:
        raise ValueError("decile must be in [1, 10]")
    q = decile / 10.0
    return _ordered_quantile(da=da, q=q, dim=member_dim)


def fraction_exceedance_mask(
    da: xr.DataArray,
    threshold: float,
    required_fraction: float,
    member_dim: str = "member",
    strict: bool = False,
) -> xr.DataArray:
    if required_fraction < 0.0 or required_fraction > 1.0:
        raise ValueError("required_fraction must be in [0, 1]")
    frac = probability_exceedance(
        da=da,
        threshold=threshold,
        member_dim=member_dim,
        strict=strict,
    )
    return frac >= required_fraction


def paintball_bitmask(
    da: xr.DataArray,
    threshold: float,
    member_dim: str = "member",
    strict: bool = False,
    output_dtype: str | np.dtype | None = None,
) -> xr.DataArray:
    if member_dim not in da.dims:
        raise ValueError(f"member_dim '{member_dim}' not found in dims {da.dims}")

    member_count = int(da.sizes[member_dim])
    if member_count < 1:
        raise ValueError("member_dim must have at least one member")

    dtype = _resolve_paintball_dtype(member_count, output_dtype)

    threshold_mask = (da > threshold) if strict else (da >= threshold)
    threshold_mask = threshold_mask.fillna(False)

    if member_dim in da.coords:
        member_values = da.coords[member_dim].values
    else:
        member_values = np.arange(member_count, dtype=np.int64)

    weights = (np.uint64(1) << np.arange(member_count, dtype=np.uint64)).astype(dtype, copy=False)
    bit_weights = xr.DataArray(
        weights,
        dims=(member_dim,),
        coords={member_dim: member_values},
    )

    packed = (threshold_mask.astype(dtype) * bit_weights).sum(dim=member_dim, skipna=False).astype(dtype)
    if da.name:
        packed.name = f"{da.name}_paintball"

    packed.attrs.update(
        {
            "diagnostic": "paintball_bitmask",
            "paintball_member_dim": member_dim,
            "paintball_member_count": member_count,
            "paintball_threshold": float(threshold),
            "paintball_strict": bool(strict),
            "paintball_dtype": str(dtype),
            "paintball_bit_order": "2**member_index where index follows member_dim coordinate order",
            "paintball_members": [str(value) for value in member_values.tolist()],
        }
    )
    return packed


def rolling_window_max(
    da: xr.DataArray,
    window: int,
    time_dim: str = "time",
    min_periods: int | None = None,
) -> xr.DataArray:
    if window < 1:
        raise ValueError("window must be >= 1")

    required = window if min_periods is None else min_periods
    return da.rolling({time_dim: window}, min_periods=required).max()


def neighborhood_max(
    da: xr.DataArray,
    radius_x: int,
    radius_y: int,
    x_dim: str = "x",
    y_dim: str = "y",
    min_periods: int = 1,
) -> xr.DataArray:
    if radius_x < 0 or radius_y < 0:
        raise ValueError("radius_x and radius_y must be >= 0")

    wx = (2 * radius_x) + 1
    wy = (2 * radius_y) + 1

    out = da.rolling({x_dim: wx}, center=True, min_periods=min_periods).max()
    out = out.rolling({y_dim: wy}, center=True, min_periods=min_periods).max()
    return out


def neighborhood_probability_exceedance(
    da: xr.DataArray,
    threshold: float,
    radius_x: int,
    radius_y: int,
    member_dim: str = "member",
    strict: bool = True,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    binary = (da > threshold) if strict else (da >= threshold)
    binary = binary.astype("float32")
    nh = neighborhood_max(binary, radius_x=radius_x, radius_y=radius_y, x_dim=x_dim, y_dim=y_dim)
    return nh.mean(dim=member_dim, skipna=True)


def neighborhood_probability_time_window(
    da: xr.DataArray,
    threshold: float,
    time_window_steps: int,
    radius_x: int,
    radius_y: int,
    member_dim: str = "member",
    time_dim: str = "time",
    x_dim: str = "x",
    y_dim: str = "y",
    strict: bool = True,
) -> xr.DataArray:
    threshold_binary = (da > threshold) if strict else (da >= threshold)
    time_agg = rolling_window_max(
        threshold_binary.astype("float32"),
        window=time_window_steps,
        time_dim=time_dim,
        min_periods=time_window_steps,
    )
    nh = neighborhood_max(
        time_agg,
        radius_x=radius_x,
        radius_y=radius_y,
        x_dim=x_dim,
        y_dim=y_dim,
    )
    return nh.mean(dim=member_dim, skipna=True)


def probability_matched_mean(
    da: xr.DataArray,
    member_dim: str = "member",
    y_dim: str | None = None,
    x_dim: str | None = None,
) -> xr.DataArray:
    if member_dim not in da.dims:
        raise ValueError(f"member_dim '{member_dim}' not found in dims {da.dims}")

    resolved_y, resolved_x = _iter_xy_dims(da, y_dim=y_dim, x_dim=x_dim)
    mean_da = da.mean(dim=member_dim, skipna=True)

    def _pmm_core(member_field: np.ndarray) -> np.ndarray:
        n_member = member_field.shape[0]
        spatial_shape = member_field.shape[1:]
        n_points = int(np.prod(spatial_shape))

        mean_flat = np.nanmean(member_field, axis=0).reshape(-1)
        all_flat = member_field.reshape(n_member * n_points)

        valid_mean = np.isfinite(mean_flat)
        valid_all = np.isfinite(all_flat)

        if valid_mean.sum() == 0 or valid_all.sum() == 0:
            return np.full(spatial_shape, np.nan, dtype=np.float32)

        mean_vals = mean_flat[valid_mean]
        mean_rank_idx = np.argsort(mean_vals)

        dist_vals = np.sort(all_flat[valid_all])

        n_mean = mean_vals.size
        n_dist = dist_vals.size

        if n_mean == 1:
            mapped = np.array([dist_vals[n_dist // 2]], dtype=np.float64)
        else:
            q = np.arange(n_mean, dtype=np.float64) / (n_mean - 1)
            d_idx = np.clip(np.round(q * (n_dist - 1)).astype(np.int64), 0, n_dist - 1)
            mapped = dist_vals[d_idx]

        mean_matched = np.full_like(mean_vals, np.nan, dtype=np.float64)
        mean_matched[mean_rank_idx] = mapped

        out_flat = np.full_like(mean_flat, np.nan, dtype=np.float64)
        out_flat[valid_mean] = mean_matched
        return out_flat.reshape(spatial_shape).astype(np.float32)

    pmm = xr.apply_ufunc(
        _pmm_core,
        da,
        input_core_dims=[[member_dim, resolved_y, resolved_x]],
        output_core_dims=[[resolved_y, resolved_x]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32],
    )

    pmm = pmm.assign_coords({k: v for k, v in mean_da.coords.items() if k in pmm.dims})
    pmm.attrs.update(mean_da.attrs)
    pmm.attrs["postprocess"] = "PMM"
    return pmm


def localized_probability_matched_mean(
    da: xr.DataArray,
    member_dim: str = "member",
    x_dim: str = "x",
    y_dim: str = "y",
    radius_x: int = 10,
    radius_y: int = 10,
) -> xr.DataArray:
    pmm = probability_matched_mean(
        da=da,
        member_dim=member_dim,
        y_dim=y_dim,
        x_dim=x_dim,
    )
    local_max = neighborhood_max(
        da.max(dim=member_dim, skipna=True),
        radius_x=radius_x,
        radius_y=radius_y,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    spread = da.std(dim=member_dim, skipna=True)
    robust_scale = _ordered_quantile(spread, 0.95, dim=x_dim)
    robust_scale = _ordered_quantile(robust_scale, 0.95, dim=y_dim)

    weight = xr.where(robust_scale > 0, spread / (robust_scale + 1.0e-6), 0.0)
    weight = xr.where(weight > 1.0, 1.0, xr.where(weight < 0.0, 0.0, weight))

    lpmm = (1.0 - weight) * pmm + weight * local_max
    lpmm.attrs.update(pmm.attrs)
    lpmm.attrs["postprocess"] = "LPMM"
    lpmm.attrs["lpmm_radius_x"] = int(radius_x)
    lpmm.attrs["lpmm_radius_y"] = int(radius_y)
    return lpmm.astype(np.float32)


def spaghetti_contour_mask(
    da: xr.DataArray,
    contour_value: float,
    member_dim: str = "member",
    tolerance: float = 0.25,
) -> xr.DataArray:
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    lower = contour_value - tolerance
    upper = contour_value + tolerance
    return ((da >= lower) & (da <= upper)).astype("uint8")


def spaghetti_probability_band(
    da: xr.DataArray,
    contour_value: float,
    member_dim: str = "member",
    tolerance: float = 0.25,
) -> xr.DataArray:
    mask = spaghetti_contour_mask(
        da=da,
        contour_value=contour_value,
        member_dim=member_dim,
        tolerance=tolerance,
    )
    return mask.mean(dim=member_dim, skipna=True)


def extract_spaghetti_contours(
    da: xr.DataArray,
    contour_value: float,
    member_dim: str = "member",
    time_dim: str = "time",
    x_coord: str = "lon",
    y_coord: str = "lat",
) -> list[dict]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for contour extraction. Install with: pip install matplotlib"
        ) from exc

    if member_dim not in da.dims:
        raise ValueError(f"member_dim '{member_dim}' not found in dims {da.dims}")
    if time_dim not in da.dims:
        raise ValueError(f"time_dim '{time_dim}' not found in dims {da.dims}")

    if x_coord not in da.coords or y_coord not in da.coords:
        raise ValueError(f"Expected coords '{x_coord}' and '{y_coord}' in DataArray coords")

    xvals = da.coords[x_coord].values
    yvals = da.coords[y_coord].values

    results: list[dict] = []
    for t in da.coords[time_dim].values:
        for m in da.coords[member_dim].values:
            fld = da.sel({time_dim: t, member_dim: m}).values
            cs = plt.contour(xvals, yvals, fld, levels=[contour_value])

            segments: list[list[list[float]]] = []
            for col in cs.collections:
                for path in col.get_paths():
                    vertices = path.vertices
                    if vertices.shape[0] < 2:
                        continue
                    segments.append(vertices.astype(np.float32).tolist())

            plt.close()
            results.append(
                {
                    "member": _coerce_scalar(m),
                    "time": _coerce_scalar(t),
                    "segments": segments,
                }
            )

    return results


def _coerce_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def add_grid_metadata_attrs(
    ds: xr.Dataset,
    *,
    y_coord_candidates: Sequence[str] = DEFAULT_Y_CANDIDATES,
    x_coord_candidates: Sequence[str] = DEFAULT_X_CANDIDATES,
    normalize_lon: bool = False,
) -> xr.Dataset:
    out = ds.copy()

    y_name = _find_coord_name(out, y_coord_candidates)
    x_name = _find_coord_name(out, x_coord_candidates)

    if y_name is None or x_name is None:
        return out

    yv = np.asarray(out[y_name].values)
    xv = np.asarray(out[x_name].values)

    if normalize_lon:
        xv = _normalize_longitudes(xv)

    if yv.ndim == 1 and xv.ndim == 1:
        dy = abs(float(yv[1] - yv[0])) if yv.size > 1 else np.nan
        dx = abs(float(xv[1] - xv[0])) if xv.size > 1 else np.nan
    else:
        dy = abs(float(yv[1, 0] - yv[0, 0])) if yv.shape[0] > 1 else np.nan
        dx = abs(float(xv[0, 1] - xv[0, 0])) if xv.shape[1] > 1 else np.nan

    out.attrs.update(
        {
            "grid_y_coord": str(y_name),
            "grid_x_coord": str(x_name),
            "grid_lat_min": float(np.nanmin(yv)),
            "grid_lat_max": float(np.nanmax(yv)),
            "grid_lon_min": float(np.nanmin(xv)),
            "grid_lon_max": float(np.nanmax(xv)),
            "grid_dy": float(dy) if np.isfinite(dy) else "unknown",
            "grid_dx": float(dx) if np.isfinite(dx) else "unknown",
        }
    )
    return out


def apply_ensemble_diagnostics(
    ds: xr.Dataset,
    variable: str,
    *,
    member_dim: str | None = None,
    time_dim: str | None = None,
    y_dim: str | None = None,
    x_dim: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    percentile_probs: Sequence[float] | None = None,
    neighborhood_probability_requests: Sequence[dict] | None = None,
    contour_band_requests: Sequence[dict] | None = None,
    paintball_requests: Sequence[dict] | None = None,
    include_pmm: bool = True,
    include_lpmm: bool = True,
    lpmm_radius_x: int = 10,
    lpmm_radius_y: int = 10,
) -> xr.Dataset:
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in dataset")

    da = ds[variable]
    dims = infer_ensemble_dims(
        da,
        member_dim=member_dim,
        time_dim=time_dim,
        y_dim=y_dim,
        x_dim=x_dim,
    )

    out = xr.Dataset()
    out[f"{variable}_mean"] = mean_field(da, member_dim=dims.member_dim)
    out[f"{variable}_spread"] = spread_field(da, member_dim=dims.member_dim)
    out[f"{variable}_max"] = max_field(da, member_dim=dims.member_dim)
    out[f"{variable}_min"] = min_field(da, member_dim=dims.member_dim)

    if thresholds:
        for label, threshold in thresholds.items():
            out[f"prob_{label}"] = probability_exceedance(
                da,
                threshold=float(threshold),
                member_dim=dims.member_dim,
                strict=True,
            )

    if percentile_probs:
        for q in percentile_probs:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"percentile q must be in [0,1], got {q}")
            pct_name = int(round(q * 100.0))
            out[f"{variable}_p{pct_name:02d}"] = _ordered_quantile(
                da,
                q=float(q),
                dim=dims.member_dim,
            )

    if neighborhood_probability_requests:
        if dims.time_dim is None:
            raise ValueError(
                "No time dimension could be inferred; set time_dim explicitly or omit neighborhood_probability_requests."
            )

        for req in neighborhood_probability_requests:
            name = str(req["name"])
            thr = float(req["threshold"])
            tw = int(req["time_window_steps"])
            rx = int(req["radius_x"])
            ry = int(req["radius_y"])
            strict = bool(req.get("strict", True))

            out[name] = neighborhood_probability_time_window(
                da=da,
                threshold=thr,
                time_window_steps=tw,
                radius_x=rx,
                radius_y=ry,
                member_dim=dims.member_dim,
                time_dim=dims.time_dim,
                x_dim=dims.x_dim,
                y_dim=dims.y_dim,
                strict=strict,
            )

    if contour_band_requests:
        for req in contour_band_requests:
            name = str(req["name"])
            contour_value = float(req["contour_value"])
            tol = float(req.get("tolerance", 0.25))
            out[name] = spaghetti_probability_band(
                da=da,
                contour_value=contour_value,
                member_dim=dims.member_dim,
                tolerance=tol,
            )

    if paintball_requests:
        for req in paintball_requests:
            name = str(req["name"])
            threshold = float(req["threshold"])
            strict = bool(req.get("strict", False))
            output_dtype = req.get("output_dtype", req.get("dtype"))
            out[name] = paintball_bitmask(
                da=da,
                threshold=threshold,
                member_dim=dims.member_dim,
                strict=strict,
                output_dtype=output_dtype,
            )

    if include_pmm:
        out[f"{variable}_pmm"] = probability_matched_mean(
            da=da,
            member_dim=dims.member_dim,
            y_dim=dims.y_dim,
            x_dim=dims.x_dim,
        )
    if include_lpmm:
        out[f"{variable}_lpmm"] = localized_probability_matched_mean(
            da=da,
            member_dim=dims.member_dim,
            x_dim=dims.x_dim,
            y_dim=dims.y_dim,
            radius_x=lpmm_radius_x,
            radius_y=lpmm_radius_y,
        )

    out.attrs.update(
        {
            "source_variable": variable,
            "member_dim": dims.member_dim,
            "time_dim": dims.time_dim,
            "x_dim": dims.x_dim,
            "y_dim": dims.y_dim,
        }
    )
    return out


def write_dataset_to_zarr(
    ds: xr.Dataset,
    zarr_path: str | Path,
    *,
    mode: str = "w",
    consolidated: bool = True,
    compute: bool = True,
) -> None:
    encoding: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, variable in ds.variables.items():
        if variable.ndim == 0:
            continue
        chunks = []
        for size in variable.shape:
            if size <= 1:
                chunks.append(1)
            else:
                chunks.append(min(size, 256))
        encoding[name] = {"chunks": tuple(chunks)}

    ds.to_zarr(
        str(zarr_path),
        mode=mode,
        consolidated=consolidated,
        zarr_format=2,
        encoding=encoding,
        compute=compute,
    )


def convert_dataset_to_zarr(
    ds: xr.Dataset,
    zarr_path: str | Path,
    *,
    include_grid_metadata: bool = True,
    normalize_lon: bool = False,
    mode: str = "w",
    consolidated: bool = True,
    compute: bool = True,
) -> xr.Dataset:
    out = ds
    if include_grid_metadata:
        out = add_grid_metadata_attrs(out, normalize_lon=normalize_lon)
    write_dataset_to_zarr(
        out,
        zarr_path=zarr_path,
        mode=mode,
        consolidated=consolidated,
        compute=compute,
    )
    return out


def process_ensemble_to_zarr(
    ds: xr.Dataset,
    variable: str,
    output_zarr_path: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
    percentile_probs: Sequence[float] | None = None,
    neighborhood_probability_requests: Sequence[dict] | None = None,
    contour_band_requests: Sequence[dict] | None = None,
    paintball_requests: Sequence[dict] | None = None,
    include_pmm: bool = True,
    include_lpmm: bool = True,
    lpmm_radius_x: int = 10,
    lpmm_radius_y: int = 10,
    include_grid_metadata: bool = True,
    normalize_lon: bool = False,
    mode: str = "w",
    consolidated: bool = True,
    compute: bool = True,
) -> xr.Dataset:
    diag = apply_ensemble_diagnostics(
        ds=ds,
        variable=variable,
        thresholds=thresholds,
        percentile_probs=percentile_probs,
        neighborhood_probability_requests=neighborhood_probability_requests,
        contour_band_requests=contour_band_requests,
        paintball_requests=paintball_requests,
        include_pmm=include_pmm,
        include_lpmm=include_lpmm,
        lpmm_radius_x=lpmm_radius_x,
        lpmm_radius_y=lpmm_radius_y,
    )

    if include_grid_metadata:
        diag = add_grid_metadata_attrs(diag, normalize_lon=normalize_lon)

    write_dataset_to_zarr(
        diag,
        zarr_path=output_zarr_path,
        mode=mode,
        consolidated=consolidated,
        compute=compute,
    )
    return diag


def example_usage() -> str:
    return (
        "import xarray as xr\n"
        "from datastore.diagnostics.ensemble import process_ensemble_to_zarr\n\n"
        "ds = xr.open_dataset('input.nc', chunks={'time': 1, 'member': 5})\n"
        "diag = process_ensemble_to_zarr(\n"
        "    ds=ds,\n"
        "    variable='uh',\n"
        "    output_zarr_path='uh_diag.zarr',\n"
        "    thresholds={'uh_gt_75': 75.0, 'stp_gt_1': 1.0},\n"
        "    percentile_probs=[0.1, 0.5, 0.9],\n"
        "    neighborhood_probability_requests=[\n"
        "        {'name': 'uh75_1h_nprob', 'threshold': 75.0, 'time_window_steps': 1, 'radius_x': 8, 'radius_y': 8},\n"
        "        {'name': 'uh75_4h_nprob', 'threshold': 75.0, 'time_window_steps': 4, 'radius_x': 8, 'radius_y': 8},\n"
        "        {'name': 'uh75_24h_nprob', 'threshold': 75.0, 'time_window_steps': 24, 'radius_x': 8, 'radius_y': 8},\n"
        "    ],\n"
        "    contour_band_requests=[\n"
        "        {'name': 'dpt70_spag_prob', 'contour_value': 70.0, 'tolerance': 0.5},\n"
        "    ],\n"
        "    paintball_requests=[\n"
        "        {'name': 'uh75_paintball', 'threshold': 75.0, 'strict': False},\n"
        "    ],\n"
        ")\n"
    )


__all__ = [
    "EnsembleDims",
    "infer_ensemble_dims",
    "mean_field",
    "spread_field",
    "max_field",
    "min_field",
    "probability_exceedance",
    "probability_in_range",
    "exceedance_fraction",
    "decile_membership",
    "fraction_exceedance_mask",
    "paintball_bitmask",
    "rolling_window_max",
    "neighborhood_max",
    "neighborhood_probability_exceedance",
    "neighborhood_probability_time_window",
    "probability_matched_mean",
    "localized_probability_matched_mean",
    "spaghetti_contour_mask",
    "spaghetti_probability_band",
    "extract_spaghetti_contours",
    "add_grid_metadata_attrs",
    "apply_ensemble_diagnostics",
    "write_dataset_to_zarr",
    "convert_dataset_to_zarr",
    "process_ensemble_to_zarr",
    "example_usage",
]