# The Pipeline

src/sources includes paths to the remote and local data sources
for ensemble cyclone tracks (ATCF files and ECMWF ensemble cyclone tracks)
as well as dynamical.org files and the NSSL GEFS datafeed.

We can also download GRIB2 data files from NSSL for the HREF members.

We can also download the GRIB2 data files using Herbie to get the ECMWF HIRES forecasts.

We should also eventually be able to download the GRIB2 datasets for the REFS.

I should also consider putting the BUFR sounding model data logic I've developed here.

We can also stage ensemble member forecasts in the case of the HREF or REFS
in zarr files from the GRIB2 directories.  For example:

~/data/zarr/2026032200.href_members/
    hrrr.zarr
    wrf4nssl.zarr
    hiresw_conusfv3.zarr
    hiresw_conusarw.zarr
    namnest.zarr
    nssl_mpashn.zarr

~/data/zarr/2026032112.href_members/
    hrrr.zarr
    wrf4nssl.zarr
    hiresw_conusfv3.zarr
    hiresw_conusarw.zarr
    namnest.zarr
    nssl_mpashn.zarr

An ensemble dataset in this application will have the dimensions: (time, member, x, y)

Once we have the necessary dataset in these dimensions, we calculate our ensemble output
including:

- mean
- spread
- event probabilities
- paintball plots
- contour extraction for spaghetti diagrams
- neighborhood probabilities
- probablity matched means
- local probability matched means
- precip type and intensity probabilities

**We should keep this separate from the logic for the observation data store**

## Lagged Ensemble Workflow For Accumulation Fields

Use this workflow when building ensembles from current and time-lagged member stores.

1. Create a variables inventory that keeps only consistently-available fields:

     python src/EnsDataStore/pipelines/create_inventory.py \
         --input-root ~/data/base/model/wrf4nssl/ \
         --output-variables href_variables.json \
         --sample-limit 30 \
         --min-file-fraction 1.0

2. Build each member store for a cycle (or latest cycle only):

     python src/EnsDataStore/pipelines/generate_memberStore.py \
         --input-root ~/data/base/model/wrf4nssl/ \
         --member wrf4nssl \
         --variables-db href_variables.json \
         --output-dir ~/data/zarr/2026032300.href_members/ \
         --latest-cycle-only

     Member stores now include per-time metadata (`source_cycle_time`,
     `source_forecast_hour`, `source_lag_hours`) and accumulation metadata
     (`is_run_accumulated`, `accumulation_hours`).

3. Postprocess member stores into an ensemble dataset with accumulation products:

     python src/EnsDataStore/pipelines/postprocess_ensemble.py \
         --input-root ~/data/zarr/2026032300.href_members/ \
         --output-path ~/data/zarr/2026032300.href_ensemble.zarr \
         --accum-grib-name APCP WEASD \
         --prefer-native-1h \
         --reconstruct-run-total

     Postprocessing prefers native 1-hour accumulation fields when available, and
     falls back to cycle-aware differencing of run-total accumulations when needed.

