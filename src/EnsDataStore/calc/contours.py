"""Contour extraction helpers for ensemble diagnostic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def extract_contours_numpy(field: np.ndarray, lons: np.ndarray, lats: np.ndarray, level_value: float) -> list[np.ndarray]:
    """Extract contour segments for one field at a single level value."""
    import matplotlib.pyplot as plt

    contour_set = plt.contour(lons, lats, field, levels=[level_value])

    contours: list[np.ndarray] = []
    for collection in contour_set.collections:
        for path in collection.get_paths():
            contours.append(path.vertices)

    plt.close()
    return contours


def contours_to_chunk(
    contours: list[np.ndarray],
    member_idx: int,
    time_idx: int,
    seg_start: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None, int]:
    """Pack contour paths into flat arrays for appending into a Zarr group."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    members: list[np.ndarray] = []
    times: list[np.ndarray] = []
    segment_ids: list[np.ndarray] = []

    seg_id = seg_start

    for contour in contours:
        if len(contour) < 2:
            continue

        x = contour[:, 0].astype("float32")
        y = contour[:, 1].astype("float32")
        n_points = len(x)

        xs.append(x)
        ys.append(y)
        members.append(np.full(n_points, member_idx, dtype="int16"))
        times.append(np.full(n_points, time_idx, dtype="int32"))
        segment_ids.append(np.full(n_points, seg_id, dtype="int32"))

        seg_id += 1

    if not xs:
        return None, seg_id

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(members),
        np.concatenate(times),
        np.concatenate(segment_ids),
    ), seg_id


def generate_contours_to_zarr(
    zarr_path: str | Path,
    var_name: str = "gh",
    pressure_level: int = 500,
    contour_value: float = 5400.0,
) -> str:
    """Extract contours from a member/time field and append them into the same store."""
    ds = xr.open_zarr(str(zarr_path), chunks={})

    da = ds[var_name]
    if "level" in da.dims:
        da = da.sel(level=pressure_level)

    lons = ds["lon"].values
    lats = ds["lat"].values

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_member: list[np.ndarray] = []
    all_time: list[np.ndarray] = []
    all_segment: list[np.ndarray] = []

    seg_counter = 0

    for time_idx, valid_time in enumerate(da.time.values):
        for member_idx, member in enumerate(da.member.values):
            _ = member
            field = da.sel(time=valid_time, member=da.member.values[member_idx]).values
            contours = extract_contours_numpy(field=field, lons=lons, lats=lats, level_value=contour_value)
            chunk, seg_counter = contours_to_chunk(contours, member_idx=member_idx, time_idx=time_idx, seg_start=seg_counter)
            if chunk is None:
                continue

            x, y, mem, tim, seg = chunk
            all_x.append(x)
            all_y.append(y)
            all_member.append(mem)
            all_time.append(tim)
            all_segment.append(seg)

    if all_x:
        x_data = np.concatenate(all_x)
        y_data = np.concatenate(all_y)
        member_data = np.concatenate(all_member)
        time_data = np.concatenate(all_time)
        segment_data = np.concatenate(all_segment)
    else:
        x_data = np.array([], dtype="float32")
        y_data = np.array([], dtype="float32")
        member_data = np.array([], dtype="int16")
        time_data = np.array([], dtype="int32")
        segment_data = np.array([], dtype="int32")

    root = zarr.open(str(zarr_path), mode="a")
    group_path = f"contours/{var_name}/{pressure_level}/{int(contour_value)}"
    group = root.require_group(group_path)

    group.create_dataset("x", data=x_data, overwrite=True, chunks=(100000,))
    group.create_dataset("y", data=y_data, overwrite=True, chunks=(100000,))
    group.create_dataset("member", data=member_data, overwrite=True, chunks=(100000,))
    group.create_dataset("time", data=time_data, overwrite=True, chunks=(100000,))
    group.create_dataset("segment_id", data=segment_data, overwrite=True, chunks=(100000,))

    return group_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate contour segments and append to Zarr store")
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--var-name", default="gh")
    parser.add_argument("--pressure-level", type=int, default=500)
    parser.add_argument("--contour-value", type=float, default=5400.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    group_path = generate_contours_to_zarr(
        zarr_path=args.zarr_path,
        var_name=args.var_name,
        pressure_level=args.pressure_level,
        contour_value=args.contour_value,
    )
    print(f"Contours written to: {group_path}")
    return 0


__all__ = [
    "contours_to_chunk",
    "extract_contours_numpy",
    "generate_contours_to_zarr",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
