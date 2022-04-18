    #from utils import load_chl
    #import os
    #import glob

    #chl_path =  os.path.join(os.getenv("DATA"),'chl_data')
    #chl_root = '*.nc'
        # Load data
    #climatologyPeriod=[2011,2011]
    #if chl_path is None:
        #chl_path =  os.path.join(os.getenv("chl_data"))
        #chl_root = '*.nc'

    #data_in=None
    #if data_in is None:
        #lat_coord, lon_coord, t, all_chl = load_chl(
            #chl_path, chl_root, climatologyPeriod)
    #else:
        #lat_coord, lon_coord, t, all_chl = data_in



import os
import glob
from typing import IO
from pkg_resources import resource_filename

import numpy as np
from scipy import signal

from datetime import date
import datetime

import pandas
import xarray

from mhw import utils as mhw_utils
from mhw import mhw_numba

from mhw_analysis.chl import chl_utils
from mhw import climate

from IPython import embed

istart=None
iend=None
def chl_thresh(climate_db_file,
                     chl_path=None,
                     climatologyPeriod=(2011, 2011),
                     cut_sky=False, 
                     data_in=None,
                     scale_file=None,
                     smoothPercentile = True,
                     pctile=90.,
                     interpolated=False,
                     detrend_local=None,
                     min_frac=0.9, n_calc=None, debug=False):
    """
    Build climate model from NOAA OI data

    Parameters
    ----------
    climate_db_file : str
        output filename.  Should have extension .nc
    noaa_path : str, optional
        Path to NOAA OI SST files
        Defults to os.getenv("NOAA_OI")
    climatologyPeriod : tuple, optional
        Range of years defining the climatology; inclusive
    cut_sky : bool, optional
        Used for debugging
    data_in : tuple, optional
        Loaded SST data.
        lat_coord, lon_coord, t, all_sst
    scale_file : str, optional
        Used to remove the overall trend in SST warming
    pctile : float, optional
        Percentile for T threshold
    smoothPercentile : bool, optional
        If True, smooth the measure in a 31 day box.  Default=True
    min_frac : float
        Minimum fraction required for analysis
    n_calc : int, optional
        Used for debugging
    debug : bool, optional
        Turn on debugging
    interpolated : bool, optional
        If True, files are interpolated
    detrend_local : str, optional
        If provided, load this file to detrend the SSTa values
    """
    # Path
    if chl_path is None:
        chl_path =  os.path.join(os.getenv("DATA"),'chl_data')
        chl_root = '*.nc'

    data_in=None
    if data_in is None:
        lat_coord, lon_coord, t, all_chl = chl_utils.load_chl(
        chl_path, chl_root, climatologyPeriod)
    else:
        lat_coord, lon_coord, t, all_chl = data_in

    print('Finsished loading')    

    # Time -- especially DOY
    time_dict = climate.build_time_dict(t)

    # Scaling
    scls = np.zeros_like(t).astype(float)
    if scale_file is not None:
        # Use scales
        if scale_file[-3:] == 'hdf':
            scale_tbl = pandas.read_hdf(scale_file, 'median_climate')
        elif scale_file[-7:] == 'parquet':
            scale_tbl = pandas.read_parquet(scale_file)
        else:
            raise IOError("Not ready for this type of scale_file: {}".format(scale_file))
        for kk, it in enumerate(t):
            mtch = np.where(scale_tbl.index.to_pydatetime() == datetime.datetime.fromordinal(it))[0][0]
            scls[kk] = scale_tbl.medSSTa_savgol[mtch]
    print('Finshed scaling')
    # De-trend
    if detrend_local is not None: 
        # Check
        if scale_file is not None:
            raise IOError("Don't mix scaling and de-trending!")
        # Load up
        ds_detrend = xarray.open_dataset(detrend_local)
        # Lazy
        _, _ = ds_detrend.slope.data[:], ds_detrend.y.data[:]

    # Start the db's
    if os.path.isfile(climate_db_file):
        print("Removing existing db_file: {}".format(climate_db_file))
        os.remove(climate_db_file)

    # Main loop
    if cut_sky:
        irange = np.arange(355, 365)
        jrange = np.arange(715,725)
    else:
        irange = np.arange(lat_coord.shape[0])
        jrange = np.arange(lon_coord.shape[0])
    ii_grid, jj_grid = np.meshgrid(irange, jrange)
    ii_grid = ii_grid.flatten()
    jj_grid = jj_grid.flatten()
    if n_calc is None:
        n_calc = len(irange) * len(jrange)

    # Init
    lenClimYear = 366  # This has to match what is in climate.py
    out_seas = np.zeros((lenClimYear, lat_coord.shape[0], lon_coord.shape[0]), dtype='float32')
    out_thresh = np.zeros((lenClimYear, lat_coord.shape[0], lon_coord.shape[0]), dtype='float32')
    out_linear = np.zeros((2, lat_coord.shape[0], lon_coord.shape[0]), dtype='float32')

    counter = 0

    # Length of climatological year
    lenClimYear = 366
    feb29 = 60
    # Window
    windowHalfWidth=5
    wHW_array = np.outer(np.ones(1000, dtype='int'), np.arange(-windowHalfWidth, windowHalfWidth + 1))

    # Inialize arrays
    thresh_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')
    seas_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')

    doyClim = time_dict['doy']
    TClim = len(doyClim)

    clim_start = 0
    clim_end = len(doyClim)
    nwHW = wHW_array.shape[1]
    print('Finshed initalizing arrays')
    # Smoothing
    smoothPercentileWidth = 31

    # Main loop
    while (counter < n_calc):
        # Init
        thresh_climYear[:] = np.nan
        seas_climYear[:] = np.nan

        ilat = ii_grid[counter]
        jlon = jj_grid[counter]
        counter += 1
    
        # Grab CHL values
        CHL = mhw_utils.grab_T(all_chl, ilat, jlon)
        frac = np.sum(np.invert(CHL.mask))/t.size
        if CHL.mask is np.bool_(False) or frac > min_frac:
            pass
        else:
            continue

        # De-trend
        if scale_file is not None:
            CHL -= scls

        if detrend_local is not None:
            # Detrend function
            f = np.poly1d((ds_detrend.slope[ilat,jlon],
                          ds_detrend.y[ilat,jlon]))
            # Detrend by DOY
            dCHLa = f(t)
            CHL -= dCHLa
            out_linear[:, ilat, jlon] = (ds_detrend.slope[ilat,jlon],
                          ds_detrend.y[ilat,jlon])

        # Work it
        mhw_numba.calc_clim(lenClimYear, feb29, doyClim, clim_start, 
                            clim_end, wHW_array, nwHW, TClim, 
                            thresh_climYear, CHL, pctile, 
                            seas_climYear)
        # Leap day
        thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
        seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]


        # Smooth if desired
        if smoothPercentile:
            thresh_climYear = mhw_utils.runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = mhw_utils.runavg(seas_climYear, smoothPercentileWidth)
    
        # Save
        out_seas[:, ilat, jlon] = seas_climYear
        out_thresh[:, ilat, jlon] = thresh_climYear

        # Cubes
        if (counter % 1000 == 0) or (counter == n_calc):
            print('count={} of {}.'.format(counter, n_calc))
            print("Saving...")
            climate.write_me(out_linear, out_seas, out_thresh, lat_coord, lon_coord, climate_db_file)
    print('Finshed going through data')
    # Final write
    climate.write_me(out_linear, out_seas, out_thresh, lat_coord, lon_coord, climate_db_file)
    print("All done!!")

if __name__ == '__main__':
    chl_thresh('testout.nc',chl_path=None,
                     climatologyPeriod=(2011, 2011),
                     cut_sky=False, 
                     data_in=None,
                     scale_file=None,
                     smoothPercentile = True,
                     pctile=90.,
                     interpolated=False,
                     detrend_local=None,
                     min_frac=0.9, n_calc=None, debug=False)
