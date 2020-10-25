""" Build the cube(s) that feeds into build"""

# imports
import numpy as np
from importlib import reload

from datetime import date

import pandas
import sqlalchemy
import xarray

#from cf_units import Unit
#import iris

from oceanpy.sst import utils as sst_utils

from IPython import embed

mhw_hdf_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_defaults.hdf'

def build_cube(outfile, mhw_events=None, ymd_end=(2019,12,31),
               ymd_start=(1982,1,1), mhw_db_file=None):
    """

    Args:
        outfile:
        mhw_events:
        ymd_end:
        ymd_start:

    Returns:

    """


    # Load event table
    if mhw_events is None:
        if mhw_db_file is None:
            # Original
            mhw_db_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_defaults.db'
        print("Loading the events from: {}".format(mhw_db_file))
        engine = sqlalchemy.create_engine('sqlite:///'+mhw_db_file)
        mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                       columns=['date', 'lon', 'lat', 'duration', 'time_peak',
                                                'ievent', 'time_start', 'index', 'category'])

    print("Events are loaded")

    # Size the cube for coords
    ilon = ((mhw_events['lon'].values - 0.125) / 0.25).astype(np.int32)
    jlat = ((mhw_events['lat'].values + 89.975) / 0.25).astype(np.int32)

    # Times
    ntimes = date(ymd_end[0], ymd_end[1], ymd_end[2]).toordinal() - date(
        ymd_start[0], ymd_start[1], ymd_start[2]).toordinal() + 1

    t0 = date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()

    # Categories
    categories = mhw_events['category'].values

    # Cube me
    cube = np.zeros((720, 1440, ntimes), dtype=np.int8)

    # Do it
    tstart = mhw_events['time_start'].values
    durs = mhw_events['duration'].values

    cube[:] = False
    for kk in range(len(mhw_events)):
        # Convenience
        # iilon, jjlat, tstart, dur = ilon[kk], jlat[kk], time_start[kk], durations[kk]
        #
        if kk % 1000000 == 0:
            print('kk = {} of {}'.format(kk, len(mhw_events)))
        cube[jlat[kk], ilon[kk], tstart[kk]-t0:tstart[kk]-t0+durs[kk]] = categories[kk]+1

    # Save as xarray.DataSet
    # Time
    t0 = date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()
    times = pandas.date_range(start=t0, periods=ntimes)

    # Space
    lat_coord, lon_coord = sst_utils.noaa_oi_coords()

    # Save
    da = xarray.DataArray(cube, coords=[lat_coord, lon_coord, times],
                          dims=['lat', 'lon', 'time'])
    ds = xarray.Dataset({'events': da})
    print("Saving..")
    ds.to_netcdf(outfile, engine='h5netcdf', encoding={'events': {'zlib': True}})
    print("Wrote: {}".format(outfile))

    # Return
    return cube

# Testing
if __name__ == '__main__':

    # Original
    if False:
        mhw_hdf_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_defaults.hdf'
        mhw_events = pandas.read_hdf(mhw_hdf_file, 'MHW_Events')
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube.nc',
               mhw_events=mhw_events)

    # Varying
    if False:
        mhw_hdf_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_vary.hdf'
        mhw_events = pandas.read_hdf(mhw_hdf_file, 'MHW_Events')
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc',
                   mhw_events=mhw_events)

    # 95 Varying
    if False:
        mhw_db_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_vary_95.db'
        #mhw_events = pandas.read_hdf(mhw_hdf_file, 'MHW_Events')
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary_95.nc',
                   mhw_db_file=mhw_db_file)

    # Cold std
    if True:
        mcs_db_file = '/home/xavier/Projects/Oceanography/MHW/db/mcs_events_allsky_defaults.db'
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MCSevent_cube.nc',
                   mhw_db_file=mcs_db_file)
