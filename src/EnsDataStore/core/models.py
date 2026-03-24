"""Core domain dataclasses for forecast ingestion and storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ForecastFile:
    path: Path
    member: str
    name: str
    cycle_time: datetime
    valid_time: datetime
    forecast_hour: int


@dataclass
class VariableInfo:
    name: str
    grib_name: str
    long_name: str
    units: str
    level_type: str
    level1: str
    level2: str
    type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "grib_name": self.grib_name,
            "long_name": self.long_name,
            "units": self.units,
            "level1": self.level1,
            "level2": self.level2,
            "level_type": self.level_type,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "VariableInfo":
        name = payload["name"]
        return cls(
            name=name,
            grib_name=payload.get("grib_name", name),
            long_name=payload.get("long_name", name),
            units=payload.get("units", ""),
            level_type=payload.get("level_type", ""),
            level1=payload.get("level1", ""),
            level2=payload.get("level2", ""),
            type=payload.get("type", "instant"),
        )


@dataclass
class GridInfo:
    ni: int
    nj: int
    lat_0: float
    lon_0: float
    lat_std: float
    ll_lat: float
    ll_lon: float
    ur_lat: float
    ur_lon: float
