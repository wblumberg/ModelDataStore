from siphon.http_util import session_manager
from siphon.catalog import TDSCatalog

url = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/HREF/%Y/%Y%m%d/catalog.xml"
session_manager.set_session_options(verify=False)

def open_tdscatalog(cycle_time):
    cat = TDSCatalog(cycle_time.strftime(url))
    return cat

def get_datasets(cat):
    return cat.datasets

## Either want to download the GRIB2 files and process them locally, or 
# we can download the GRIB2 file into memory and process it there.
# we're converting these to zarr files.
# either way, Siphon can help us know what files are available and download the data


