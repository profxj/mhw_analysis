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

  * mhw_events_allsky_2019.db -- SQLite format 3 table of MHW Events with columns:
    *   'time_start'           Start time of MHW [datetime format]
    *   'time_end'             End time of MHW [datetime format]
    *   'time_peak'            Time of MHW peak [datetime format]
    *   'date'                 Start date of MHW [datetime format]
    *   'index_start'          Start index of MHW
    *   'index_end'            End index of MHW
    *   'index_peak'           Index of MHW peak
    *   'duration'             Duration of MHW [days]
    *   'duration_moderate'    Duration of MHW at moderate level [days]
    *   'duration_strong'      Duration of MHW at strong level [days]
    *   'duration_severe'      Duration of MHW at severe level [days]
    *   'duration_extreme'     Duration of MHW at extreme level [days]
    *   'intensity_max'        Maximum (peak) intensity [deg. C]
    *   'intensity_mean'       Mean intensity [deg. C]
    *   'intensity_var'        Intensity variability [deg. C]
    *   'intensity_cumulative' Cumulative intensity [deg. C x days]
    *   'rate_onset'           Onset rate of MHW [deg. C / days]
    *   'rate_decline'         Decline rate of MHW [deg. C / days]
    *   'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh',
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology
    *   'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
        'intensity_cumulative_abs' are as above except as absolute magnitudes
        rather than relative to the seasonal climatology or threshold
    *   'category' is an integer category system (1, 2, 3, 4) based on the maximum intensity
        in multiples of threshold exceedances, i.e., a value of 1 indicates the MHW
        intensity (relative to the climatology) was >=1 times the value of the threshold (but
        less than 2 times; relative to climatology, i.e., threshold - climatology).
        Category types are defined as 1=strong, 2=moderate, 3=severe, 4=extreme. More details in
        Hobday et al. (in prep., Oceanography). Also supplied are the duration of each of these
        categories for each event.
    *   'n_events'             A scalar integer (not a list) indicating the total number of detected MHW events
  * mhw_events_allsky_2019.parquet -- Pandas table of MHW Events
  * MHWevent_cube_2019.nc -- netCDF file describing the location of every MHW Event in space and time.  The coordinates are: `lat (deg), lon (deg), time (datetime64[ns])`

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
