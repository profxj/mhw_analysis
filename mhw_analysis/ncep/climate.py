""" Biuld climate"""

import os
import glob
from pkg_resources import resource_filename

import numpy as np
from scipy import signal

from datetime import date
import datetime

import pandas
import iris

from mhw import climate as mhw_climate
from mhw import utils as mhw_utils
from mhw import mhw_numba


from IPython import embed


def ncep_seas_thresh(climate_db_file,
                     ncep_path=None,
                     climatologyPeriod=(1983, 2019),
                     cut_sky=True, all_sst=None,
                     scale_file=None,
                     min_frac=0.9, n_calc=None, debug=False):
    """
    Build climate model for NCEP Z500 data

    Parameters
    ----------
    climate_db_file : str
        output filename.  Should have extension .nc
    climatologyPeriod
    cut_sky
    all_sst
    min_frac
    n_calc
    debug

    Returns
    -------

    """
    # Path
    if ncep_path is None:
        ncep_path = os.getenv("NCEP_DOE")
        if ncep_path is None:
            raise IOError("Need to set NCEP_DOE")

    # Grab the cube
    ifile = os.path.join(ncep_path, 'NCEP-DOE_Z500.nc')
    cube = iris.load(ifile)[0]

    # Load the Cubes into memory
    _ = cube.data[:]

    # Coords
    lat_coord = cube.coord('latitude')
    lon_coord = cube.coord('longitude')

    # Time
    t = cube.coord('time').points  #mhw_utils.grab_t(all_sst)
    time_dict = mhw_climate.build_time_dict(t)

    # Scaling
    scls = np.zeros_like(t).astype(float)
    #if scale_file is not None:
    #    # Use scales
    #    scale_tbl = pandas.read_hdf(scale_file, 'median_climate')
    #    for kk, it in enumerate(t):
    #        mtch = np.where(scale_tbl.index.to_pydatetime() == datetime.datetime.fromordinal(it))[0][0]
    #        scls[kk] = scale_tbl.medSSTa_savgol[mtch]


    # Start the db's
    if os.path.isfile(climate_db_file):
        os.remove(climate_db_file)

    # Main loop
    if cut_sky:
        irange = np.arange(35, 39)
        jrange = np.arange(71,75)
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

    # Init climate items

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

    # Smoothing
    smoothPercentile = True
    smoothPercentileWidth = 31
    pctile = 90

    # Main loop
    while (counter < n_calc):
        # Init
        thresh_climYear[:] = np.nan
        seas_climYear[:] = np.nan

        ilat = ii_grid[counter]
        jlon = jj_grid[counter]
        counter += 1

        # Grab SST values
        Z500 = cube.data[:,ilat, jlon] # mhw_utils.grab_T(all_sst, ilat, jlon)
        #frac = np.sum(np.invert(SST.mask))/t.size
        #if SST.mask is np.bool_(False) or frac > min_frac:
        #    pass
        #else:
        #    continue

        # Work it
        Z500 -= scls
        mhw_numba.calc_clim(lenClimYear, feb29, doyClim, clim_start, clim_end, wHW_array, nwHW,
                     TClim, thresh_climYear, Z500, pctile, seas_climYear)
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
        if (counter % 100000 == 0) or (counter == n_calc):
            print('count={} of {}.'.format(counter, n_calc))
            print("Saving...")
            cubes = iris.cube.CubeList()
            time_coord = iris.coords.DimCoord(np.arange(lenClimYear), units='day', var_name='day')
            cube_seas = iris.cube.Cube(out_seas, units='m', var_name='seasonalZ500',
                                       dim_coords_and_dims=[(time_coord, 0),
                                                            (lat_coord, 1),
                                                            (lon_coord, 2)])
            cube_thresh = iris.cube.Cube(out_thresh, units='m', var_name='threshZ500',
                                         dim_coords_and_dims=[(time_coord, 0),
                                                              (lat_coord, 1),
                                                              (lon_coord, 2)])
            cubes.append(cube_seas)
            cubes.append(cube_thresh)
            # Write
            iris.save(cubes, climate_db_file, zlib=True)
            print("Wrote: {}".format(climate_db_file))

    print("All done!!")


# Command line execution
if __name__ == '__main__':

    # NCEP
    ncep_seas_thresh('NCEP-DOE_Z500_climate.nc', cut_sky=False)
