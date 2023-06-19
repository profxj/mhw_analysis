# Marine Heat Wave Systems
---

## Abstract

This archive contains dataa and tables for the analyses related to 
marine heat wave systems (MHWS).
Results of our investigations are reported in full in
"The rapid rise of severe marine heat wave systems" by
[Prochaska, Beaulieu, & Giamalki 2023](https://iopscience.iop.org/article/10.1088/2752-5295/accd0e)
(Environmental Research: Climate, Volume 2, Issue 2) 

In brief, we analyzed sea surface temperature (SST) data for the past ~30 years
to identify and characterize MHWSs.  We find that the severe MHWSs
drive the extrema of SST in space and time.

All of the data provided here 
are products of our own analysis.

## Description of the data and file structure

There are three main datasets: (1) tables and data of MHW events, built using code
from [the JXP fork](https://github.com/profxj/marineHeatWaves)
of Eric Oliver's [marineHeatWaves](https://github.com/ecjoliver/marineHeatWaves) repository;
(2) tables and netCDF files of MHWS and their masks;
and
(3) small netCDF files of MHWS analysis products. 
We describe each in turn.

### MWH Event data

  * mhw_events_allsky_2019.db -- SQLite format 3 table of MHW Events
  * mhw_events_allsky_2019.parquet -- Pandas table of MHW Events
  * MHWevent_cube_2019.nc -- netCDF file describing the location of every MHW Event in space and time.  The coordinates are: `lat (deg), lon (deg), time (datetime64[ns])`

See the [marineHeatWaves](https://github.com/ecjoliver/marineHeatWaves) repository
for a description of the columns
### MWHS data

  * MHWS_xxx.csv -- CSV table of the MHWS
  * MHWS_xxx_mask.nc -- netCDF file describing the location of every MHWS in space and time.

There are 3 pairs of these data files with xxx depending on the climatology:

  * defaults -- Uses the climatology of [Hobday et al. 2016](https://www.sciencedirect.com/science/article/pii/S0079661116000057) 
  * 2019 -- The default climatology of our paper (includes up to and including 2019)
  * local -- The 2019 climatology but with a local linear fit to remove any long term trend. 

### Analysis products

Three netCDF files that describe the locations of MHWS categorized
by their volume:

  * minor_km_dy_by_yr_2019.nc
  * moderate_km_dy_by_yr_2019.nc
  * severe_km_dy_by_yr_2019.nc

## Code/Software

All code related to this project may be found on 
[GitHub](https://github.com/profxj/mhw_analysis)
and one may cite [this DOI](https://zenodo.org/badge/latestdoi/262816104).
