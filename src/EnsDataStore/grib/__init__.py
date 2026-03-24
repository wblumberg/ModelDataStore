"""GRIB discovery, inventory, and grid helpers."""

from EnsDataStore.grib.discovery import ForecastFile, build_file_index, discover_files, match_by_valid_time, parse_filename

try:
    from EnsDataStore.grib.grid import GridInfo, extract_grid_info
except ImportError:
    GridInfo = None
    extract_grid_info = None

try:
    from EnsDataStore.grib.inventory import VariableInfo, get_variable_type, inventory_variables, load_variables_db, save_variables_db
except ImportError:
    VariableInfo = None
    get_variable_type = None
    inventory_variables = None
    load_variables_db = None
    save_variables_db = None

__all__ = [
    "ForecastFile",
    "GridInfo",
    "VariableInfo",
    "build_file_index",
    "discover_files",
    "extract_grid_info",
    "get_variable_type",
    "inventory_variables",
    "load_variables_db",
    "match_by_valid_time",
    "parse_filename",
    "save_variables_db",
]
