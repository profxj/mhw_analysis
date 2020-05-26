""" Module to generate a *large* Cube of MHW events"""
import glob
import numpy as np

import pandas
import sqlalchemy
from datetime import date

from IPython import embed
import os
import iris

from mhw_analysis.db import utils
from mhw import climate
from mhw import utils as mhw_utils


def build_noaa(climate_db_file, noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/',
               climatologyPeriod=(1983, 2012), cut_sky=True, all_sst=None,
               min_frac=0.9, n_calc=None):
    """
    Build the climate models for NOAA

    Args:
        dbfile:
        noaa_path:
        climate_db (str):
        cut_years:
        cut_sky:
        all_sst:
        nproc:
        min_frac:

    Returns:

    """
    # Grab the list of SST V2 files
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()
    # Cut on years
    if '1981' not in all_sst_files[0]:
        raise ValueError("Years not in sync!!")

    # Load the Cubes into memory
    if all_sst is None:
        istart = climatologyPeriod[0] - 1981
        iend = climatologyPeriod[1] - 1981 + 1
        all_sst_files = all_sst_files[istart:iend]

        print("Loading up the files. Be patient...")
        all_sst = utils.load_noaa_sst(all_sst_files)

    # Coords
    lat_coord = all_sst[0].coord('latitude')
    lon_coord = all_sst[0].coord('longitude')

    # Time
    t = utils.grab_t(all_sst)
    time_dict = climate.build_time_dict(t)

    # Start the db's
    if os.path.isfile(climate_db_file):
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

    counter = 0
    tot_events = 0

    # Init climate

    # Length of climatological year
    lenClimYear = 366
    feb29 = 60
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

    smoothPercentile = True
    smoothPercentileWidth = 31
    pctile = 90

    while (counter < n_calc):
        # Load Temperatures
        #list_SSTs, ilats, jlons = [], [], []
        thresh_climYear[:] = np.nan
        seas_climYear[:] = np.nan
        nmask = 0

        ilat = ii_grid[counter]
        jlon = jj_grid[counter]
        counter += 1
        #
        SST = utils.grab_T(all_sst, ilat, jlon)
        frac = np.sum(np.invert(SST.mask))/t.size
        if SST.mask is np.bool_(False) or frac > min_frac:
            pass
        else:
            nmask += 1
            continue
        # Work it
        climate.doit(lenClimYear, feb29, doyClim, clim_start, clim_end, wHW_array, nwHW,
                     TClim, thresh_climYear, SST, pctile, seas_climYear)
        thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
        seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]


        # Smooth if desired
        if smoothPercentile:
            thresh_climYear = mhw_utils.runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = mhw_utils.runavg(seas_climYear, smoothPercentileWidth)

        # Save
        out_seas[:, ilat, jlon] = seas_climYear
        out_thresh[:, ilat, jlon] = thresh_climYear

        # Count
        print('count={} of {}.'.format(counter, n_calc))

        # Cubes
        if (counter % 100000 == 0) or (counter == n_calc):
            print("Saving...")
            cubes = iris.cube.CubeList()
            time_coord = iris.coords.DimCoord(np.arange(lenClimYear), units='day', var_name='day')
            cube_seas = iris.cube.Cube(out_seas, units='C', var_name='seasonalT',
                                             dim_coords_and_dims=[(time_coord, 0),
                                                                  (lat_coord, 1),
                                                                  (lon_coord, 2)])
            cube_thresh = iris.cube.Cube(out_thresh, units='C', var_name='threshT',
                                       dim_coords_and_dims=[(time_coord, 0),
                                                            (lat_coord, 1),
                                                            (lon_coord, 2)])
            cubes.append(cube_seas)
            cubes.append(cube_thresh)
            # Write
            iris.save(cubes, climate_db_file, zlib=True)

    print("All done!!")


