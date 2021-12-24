""" Build the cube(s) that feeds into build systems"""

# imports
import os
import numpy as np
from importlib import reload
import pathlib

from datetime import date

import pandas
import sqlalchemy
import xarray

#from cf_units import Unit
#import iris

from oceanpy.sst import utils as sst_utils

from IPython import embed

mhw_path = os.getenv('MHW')
mhwdb_path = os.path.join(mhw_path, 'db')

def build_cube(outfile, mhw_events=None, ymd_end=(2019,12,31),
               ymd_start=(1982,1,1), mhw_db_file=None, 
               lat_lon_coord=None, test=False,
               angular_res=0.25, lon_lat_min=(0.125, -89.975)):
    """

    Args:
        outfile:
        mhw_events:
        ymd_end:
        ymd_start:
        angular_res : float, optional
            Angular resolution in deg
        lon_lat_min : tuple, optional
            Minimum lon, lat values in deg
        lat_lon_coord : tuple, optional
            xarray.Coords
        test : bool, optional
            For testing, debugging

    Returns:

    """


    # Load event table
    if mhw_events is None:
        if mhw_db_file is None:
            raise IOError("Need to provide a file")
        print("Loading the events from: {}".format(mhw_db_file))
        if pathlib.Path(mhw_db_file).suffix == '.parquet':
            mhw_events = pandas.read_parquet(mhw_db_file)
        elif pathlib.Path(mhw_db_file).suffix == '.db':
            engine = sqlalchemy.create_engine('sqlite:///'+mhw_db_file)
            mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                       columns=['date', 'lon', 'lat', 'duration', 'time_peak',
                                                'ievent', 'time_start', 'index', 'category'])
        else:
            raise IOError("Not ready for this file type")

    print("Events are loaded")

    # Size the cube for coords
    ilon = ((mhw_events['lon'].values - lon_lat_min[0]) / angular_res).astype(np.int32)
    jlat = ((mhw_events['lat'].values - lon_lat_min[1]) / angular_res).astype(np.int32)

    # Times
    ntimes = date(ymd_end[0], ymd_end[1], ymd_end[2]).toordinal() - date(
        ymd_start[0], ymd_start[1], ymd_start[2]).toordinal() + 1

    t0 = date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()

    # Categories
    categories = mhw_events['category'].values

    # Cube me
    if angular_res == 0.25:
        idim, jdim = 720, 1440
    elif test:
        idim, jdim = 21, 29
    elif angular_res == 2.5:
        idim, jdim = 72, 144
    else:
        raise IOError("Not ready for this ang_res")
    cube = np.zeros((idim, jdim, ntimes), dtype=np.int8)

    # Do it
    tstart = mhw_events['time_start'].values
    durs = mhw_events['duration'].values

    cube[:] = False
    for kk in range(len(mhw_events)):
        if kk % 1000000 == 0:
            print('kk = {} of {}'.format(kk, len(mhw_events)))
        cube[jlat[kk], ilon[kk], tstart[kk]-t0:tstart[kk]-t0+durs[kk]] = categories[kk]+1

    # Save as xarray.DataSet
    # Time
    pt0 = date(ymd_start[0], ymd_start[1], ymd_start[2])
    times = pandas.date_range(start=pt0, periods=ntimes)

    # Space
    if lat_lon_coord is None:
        lat_coord, lon_coord = sst_utils.noaa_oi_coords()
    else:
        lat_coord, lon_coord = lat_lon_coord

    # Save
    da = xarray.DataArray(cube, coords=[lat_coord, lon_coord, times],
                          dims=['lat', 'lon', 'time'])
    ds = xarray.Dataset({'events': da})
    print("Saving..")
    ds.to_netcdf(outfile, engine='h5netcdf', encoding={'events': {'zlib': True}})
    print("Wrote: {}".format(outfile))

    # Return
    return cube

def main(flg_main):
    if flg_main == 'all':
        flg_main = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_main = int(flg_main)
        
    noaa_path = os.getenv('NOAA_OI')


    # Hobday (1983,2012 climatology)
    if flg_main & (2 ** 1):
        mhw_pq_file = os.path.join(mhwdb_path, 'mhw_events_allsky_defaults.parquet')
        outfile = os.path.join(mhwdb_path, 'MHWevent_cube_defaults.nc')
        build_cube(outfile, mhw_db_file=mhw_pq_file)

    # Varying
    if flg_main & (2 ** 2):
        mhw_hdf_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_vary.db'
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc',
                   mhw_db_file=mhw_hdf_file)

    # 95 Varying
    if flg_main & (2 ** 3):
        mhw_db_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_vary_95.db'
        #mhw_events = pandas.read_hdf(mhw_hdf_file, 'MHW_Events')
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary_95.nc',
                   mhw_db_file=mhw_db_file)

    # Cold std
    if flg_main & (2 ** 4):
        mcs_db_file = '/home/xavier/Projects/Oceanography/MHW/db/mcs_events_allsky_defaults.db'
        build_cube('/home/xavier/Projects/Oceanography/MHW/db/MCSevent_cube.nc',
                   mhw_db_file=mcs_db_file)


    # Interpolated
    if flg_main & (2 ** 5):
        noaa_path = os.getenv("NOAA_OI")
        MHW_path = os.getenv("MHW")
        ds = xarray.open_dataset(os.path.join(noaa_path, 'Interpolated',
                                              'interpolated_sst_1983.nc'))
        lat_lon_coord = ds.lat, ds.lon

        # 2012
        mhw_db_file = os.path.join(MHW_path, 'db', 'mhw_events_interp2.5_2012.db')
        build_cube(os.path.join(MHW_path, 'db', 'MHWevent_cube_interp2.5_2012.nc'),
               mhw_db_file=mhw_db_file,
               lat_lon_coord=lat_lon_coord,
               angular_res=2.5, 
               lon_lat_min=(float(ds.lon.min()), float(ds.lat.min())))
               #lon_lat_min=(187.65, 12.625))

        mhw_db_file = os.path.join(MHW_path, 'db', 'mhw_events_interp2.5_2019.db')
        build_cube(os.path.join(MHW_path, 'db','MHWevent_cube_interp2.5_2019.nc'),
            mhw_db_file=mhw_db_file,
            lat_lon_coord=lat_lon_coord,
            angular_res=2.5, 
            lon_lat_min=(float(ds.lon.min()), float(ds.lat.min())))
            #lon_lat_min=(187.65, 12.625))

    # 2019, de-trend local
    if flg_main & (2 ** 6):
        mhw_pq_file = os.path.join(mhwdb_path, 'mhw_events_allsky_2019_local.parquet')
        outfile = os.path.join(mhwdb_path, 'MHWevent_cube_2019_local.nc')
        build_cube(outfile, mhw_db_file=mhw_pq_file)

    # 2019
    if flg_main & (2 ** 7):
        mhw_pq_file = os.path.join(mhwdb_path, 'mhw_events_allsky_2019.parquet')
        outfile = os.path.join(mhwdb_path, 'MHWevent_cube_2019.nc')
        build_cube(outfile, mhw_db_file=mhw_pq_file)



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg_main = 0
        #flg_main += 2 ** 1  # Hobday
        #flg_main += 2 ** 2  # de-trend median
        #flg_main += 2 ** 3  # T95
        #flg_main += 2 ** 4  # Cold waves
        #flg_main += 2 ** 5  # Interpolated
        #flg_main += 2 ** 6  # 2019, de-trend local
        flg_main += 2 ** 7  # 2019
    else:
        flg_main = sys.argv[1]

    main(flg_main)