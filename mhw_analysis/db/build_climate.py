""" Module to generate a *large* Cube of MHW events"""
import glob
import numpy as np
import multiprocessing

import pandas
import sqlalchemy
from datetime import date

from IPython import embed
import os
import iris

from mhw_analysis.db import utils
from mhw import climate


def build_noaa(climate_db_file, noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/',
               climatologyPeriod=(1983, 2012), cut_sky=True, all_sst=None, nproc=16, min_frac=0.9,
             n_calc=None, save_climate=False):
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
    #events_coord = iris.coords.DimCoord(np.arange(100), var_name='events')

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
    out_seas = np.zeros((t.size, irange.size, jrange.size), dtype='float32')
    out_thresh = np.zeros((t.size, irange.size, jrange.size), dtype='float32')

    counter = 0
    tot_events = 0
    pool = multiprocessing.Pool(processes=nproc)
    while (counter < n_calc):
        # Load Temperatures
        list_SSTs, ilats, jlons = [], [], []
        nmask = 0
        for ss in range(nproc):
            if counter == n_calc:
                break
            ilat = ii_grid[counter]
            jlon = jj_grid[counter]
            counter += 1
            # Ice/land??
            SST = utils.grab_T(all_sst, ilat, jlon)
            frac = np.sum(np.invert(SST.mask))/t.size
            if SST.mask is np.bool_(False) or frac > min_frac:
                list_SSTs.append(SST)
                ilats.append(ilat)
                jlons.append(jlon)
            else:
                nmask += 1
                continue
        # Detect
        #if len(list_SSTs) > 0:
        #    import pdb; pdb.set_trace()
        results = [pool.apply(climate.calc(), args=(time_dict, SSTs)) for SSTs in list_SSTs]
        final_tbl = None
        sub_events = 0
        for iilat, jjlon, clim in zip(ilats, jlons, results):
            out_seas[:, iilat, jjlon] = clim['seas']
            out_thresh[:, iilat, jjlon] = clim['thresh']

        # Count
        print('count={} of {}. {} were masked. {} MHW sub-events. {} total'.format(
            counter, n_calc, nmask, sub_events, tot_events))
        #print('lat={}, lon={}, nevent={}'.format(lat_coord[ilat].points[0], lon_coord[jlon].points[0],
        #                                         mhws['n_events']))
        # Save the dict
        #all_mhw.append(mhws)

    # Cubes
    embed(header='131 of climate')
    cubes = iris.cube.CubeList()
    for ss, key in enumerate(out_dict.keys()):
        cube = iris.cube.Cube(out_dict[key], units=units[ss], var_name=key,
                                     dim_coords_and_dims=[(lat_coord, 0),
                                                          (lon_coord, 1),
                                                          (events_coord, 2)])
        cubes.append(cube)
    # Write
    iris.save(cubes, outfile, zlib=True)

    print("All done!!")


