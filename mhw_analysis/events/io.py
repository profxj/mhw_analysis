""" I/O for MHW Events"""
import os
import numpy as np

import datetime

import iris
from cf_units import Unit
from oceanpy.sst import utils as sst_utils

def load_event_cube(mhw_ev_cube_file=None):
    #
    if mhw_ev_cube_file is None:
        mhw_ev_cube_file = os.path.join(os.getenv('MHW'), 'db', 'MHWevent_cube.nc')

    '''
    # Load
    ecube = np.load(mhw_ev_cube_file)['cube'].astype(np.int8)

    # Time
    ymd_start = (1982, 1, 1)
    t0 = datetime.date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()
    ymd_end = (2019,12,31)
    t1 = datetime.date(ymd_end[0], ymd_end[1], ymd_end[2]).toordinal()

    tunit = Unit('days since 01-01-01 00:00:00', calendar='gregorian')
    times = np.arange(t0, t1+1)
    time_coord = iris.coords.DimCoord(times, standard_name='time', units=tunit)

    # Space
    lat_coord, lon_coord = sst_utils.noaa_oi_coords(as_iris_coord=True)

    # Iris
    ecube = iris.cube.Cube(ecube, var_name='Events',
                           dim_coords_and_dims=[
                               (lat_coord, 0),
                               (lon_coord, 1),
                               (time_coord, 2), ])
    '''
    # Iris
    ecube = iris.load_cube(mhw_ev_cube_file)

    # Return
    return ecube

