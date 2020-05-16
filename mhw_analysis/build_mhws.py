""" Module to generate a *large* Cube of MHW events"""
import glob
import numpy as np
import multiprocessing

import pandas
import iris
import sqlalchemy

import marineHeatWaves as mhw

from IPython import embed
import os

def load_all_sst(sst_files):
    all_sst = []
    for ifile in sst_files:
        print(ifile)  # For progress
        cubes = iris.load(ifile)
        sst = cubes[0]
        # Get out of lazy
        _ = sst.data
        # Append
        all_sst.append(sst)
    #
    return all_sst

def grab_t(sst_list):
    allts = []
    for sst in sst_list:
        allts += (sst.coord('time').points + 657072).astype(int).tolist()  # 1880?
    return np.array(allts)

def grab_T(sst_list, i, j):
    allTs = []
    for sst in sst_list:
        allTs += [sst.data[:,i,j]]
    return np.ma.concatenate(allTs)

def build_me(dbfile, noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/',
             climate_db='/home/xavier/Projects/Oceanography/MHWs/db/climate_OI.db',
             years=[1986,1990], cut_sky=True, all_sst=None, nproc=16, min_frac=0.9,
             n_calc=None, save_climate=False):
    """
    Build the grid

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
        istart = years[0] - 1981
        iend = years[1] - 1981 + 1
        all_sst_files = all_sst_files[istart:iend]

        print("Loading up the files. Be patient...")
        all_sst = load_all_sst(all_sst_files)

    # Coords
    lat_coord = all_sst[0].coord('latitude')
    lon_coord = all_sst[0].coord('longitude')
    #events_coord = iris.coords.DimCoord(np.arange(100), var_name='events')

    # Time
    t = grab_t(all_sst)
    nmax = len(t)

    # Setup for output
    # ints -- all are days
    int_keys = ['time_start', 'time_end', 'time_peak', 'duration', 'duration_moderate', 'duration_strong',
                'duration_severe', 'duration_extreme']
    float_keys = ['intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative']
    str_keys = ['category']
    for key in float_keys.copy():
        float_keys += [key+'_relThresh', key+'_abs']
    float_keys += ['rate_onset', 'rate_decline']
    #units = ['day']*len(int_keys)

    #out_dict = {}
    #for key in int_keys:
    #    out_dict[key] = np.ma.zeros((lat_coord.shape[0], lon_coord.shape[0], 100), dtype=np.int32, fill_value=-1)

    # Start the db's
    if os.path.isfile(dbfile):
        os.remove(dbfile)
    engine = sqlalchemy.create_engine('sqlite:///'+dbfile)

    if os.path.isfile(climate_db):
        embed(header='101 of build')
    elif save_climate:
        engine_clim = sqlalchemy.create_engine('sqlite:///'+climate_db)

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

    counter = 0
    pool = multiprocessing.Pool(processes=nproc)
    if len(all_sst_files) < 30:
        climatologyPeriod=years
    else:
        climatologyPeriod=[1983,2012]
    while (counter < n_calc):
        # Load Temperatures
        list_SSTs = []
        for ss in range(nproc):
            if counter == n_calc:
                break
            ilat = ii_grid[counter]
            jlon = jj_grid[counter]
            counter += 1
            # Ice/land??
            SST = grab_T(all_sst, ilat, jlon)
            frac = np.sum(np.invert(SST.mask))/t.size
            if SST.mask is np.bool_(False) or frac > min_frac:
                list_SSTs.append(SST)
            else:
                continue
        # Detect
        #if len(list_SSTs) > 0:
        #    import pdb; pdb.set_trace()
        results = [pool.apply(mhw.detect, args=(t, SSTs, climatologyPeriod)) for SSTs in list_SSTs]
        final_tbl = None
        for result in results:
            mhws, clim = result
            # Fill me in
            nevent = mhws['n_events']
            if nevent > 0:
                int_dict = {}
                for key in int_keys:
                    int_dict[key] = mhws[key]
                # Sub table
                # Ints first
                sub_tbl = pandas.DataFrame.from_dict(int_dict)
                # Recast
                sub_tbl = sub_tbl.astype('int32')
                # Add strings
                for key in str_keys:
                    sub_tbl[key] = mhws[key]
                # Lat, lon
                sub_tbl['lat'] = lat_coord[ilat].points[0]
                sub_tbl['lon'] = lon_coord[jlon].points[0]
                # Floats
                float_dict = {}
                for key in float_keys:
                    float_dict[key] = mhws[key]
                sub_tbl2 = pandas.DataFrame.from_dict(float_dict)
                sub_tbl2 = sub_tbl2.astype('float32')
                # Final
                cat = pandas.concat([sub_tbl, sub_tbl2], axis=1, join='inner')
                if final_tbl is None:
                    final_tbl = cat
                else:
                    final_tbl.append(cat)

                if save_climate:
                # Climate
                    sub_clim = pandas.DataFrame.from_dict(clim)
                    sub_clim['lat'] = lat_coord[ilat].points[0]
                    sub_clim['lon'] = lon_coord[jlon].points[0]
                    # Add to DB
                    sub_clim.to_sql('Climatology', con=engine_clim, if_exists='append')

        # Add to DB
        if final_tbl is not None:
            final_tbl.to_sql('MHW_Events', con=engine, if_exists='append')

        # Count
        print('count={} of {}'.format(counter, n_calc))
        #print('lat={}, lon={}, nevent={}'.format(lat_coord[ilat].points[0], lon_coord[jlon].points[0],
        #                                         mhws['n_events']))
        # Save the dict
        #all_mhw.append(mhws)

    '''
    # Cubes
    cubes = iris.cube.CubeList()
    for ss, key in enumerate(out_dict.keys()):
        cube = iris.cube.Cube(out_dict[key], units=units[ss], var_name=key,
                                     dim_coords_and_dims=[(lat_coord, 0),
                                                          (lon_coord, 1),
                                                          (events_coord, 2)])
        cubes.append(cube)
    # Write
    iris.save(cubes, outfile, zlib=True)
    '''

    print("All done!!")


def dont_run_this():
    # Test from Ipython
    from importlib import reload
    import glob
    from mhw_analysis import build_mhws
    noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/'
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()
    years = [1982,2016]
    istart = years[0]-1981
    iend = years[1]-1981+1
    all_sst_files = all_sst_files[istart:iend]
    all_sst = build_mhws.load_all_sst(all_sst_files)
    build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', cut_sky=False, all_sst=all_sst, nproc=50, n_calc=1000)

# Command line execution
if __name__ == '__main__':
    #
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.db', cut_sky=True)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws_allsky.db', cut_years=True, cut_sky=False)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', years=[1982,2016], cut_sky=False, nproc=50, n_calc=1000)

    # Default run to match Oliver
    build_me('/home/xavier/Projects/Oceanography/MHWs/mhws_allsky_defaults.db', years=[1982,2016], cut_sky=False, nproc=50)

