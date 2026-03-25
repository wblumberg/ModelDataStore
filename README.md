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
```
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
```
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

```
     python src/EnsDataStore/pipelines/create_inventory.py \
         --input-root ~/data/base/model/ \
         --output-variables href_variables.json \
         --sample-limit 30 \
         --min-file-fraction 1.0
```
2. Build each member store for a cycle (or latest cycle only):
```
     python generate_memberStore.py \
         --input-root ../../../data/grib2 \
         --exclude-dirs ../../../data/grib2/ecmwf_hr \
         --member hrrr \
         --variables-db href_variables.json \
         --output-dir ../../../data/zarr/2026032112.href_variables/ \
         --cycle-time 2026032112
```   
3. Build the HREF statistics:
```
     python build_href.py \
         --current  data/zarr/2026032200.href_variables
         --lagged   data/zarr/2026032112.href_variables
         --output   data/zarr/2026032200.href_ensemble.zarr
         --run-id   202603220
```
