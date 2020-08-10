""" I/O for MHW Systems"""
import os
import datetime
import numpy as np

import pandas
import h5py

from cf_units import Unit
import iris

from oceanpy.sst import utils as sst_utils


def load_systems(mhw_sys_file=None):
    """
    Load up the MHW Systems into a pandas table

    Args:
        mhw_sys_file:

    Returns:
        pandas.DataFrame:

    """
    if mhw_sys_file is None:
        mhw_sys_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_systems.hdf')
    # Read
    mhw_sys = pandas.read_hdf(mhw_sys_file)
    # Return
    return mhw_sys


def load_mask_from_dates(ymd_start, ymd_end,
                         mhw_mask_file=None, mask_start=(1982,1,1)):
    if mhw_mask_file is None:
        mhw_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask.hdf')

    # Convert ymd to indices
    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()

    ts = datetime.date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()
    te = datetime.date(ymd_end[0], ymd_end[1], ymd_end[2]).toordinal()

    i0 = ts-t0
    i1 = te-t0+1  # Make it inclusive

    # Load + return
    return maskcube_from_slice(i0, i1, mhw_mask_file=mhw_mask_file, mask_start=mask_start)


def load_mask_from_system(mhw_system,
                          mhw_mask_file = None, mask_start = (1982, 1, 1)):
    # Load + return
    return maskcube_from_slice(mhw_system.zboxmin, mhw_system.zboxmax,
                               mhw_mask_file=mhw_mask_file, mask_start=mask_start)


def maskcube_from_slice(i0,i1,
                        mhw_mask_file=None, mask_start=(1982, 1, 1)):
    if mhw_mask_file is None:
        mhw_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask.hdf')

    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()
    ts = t0 + i0
    te = t0 + i1

    # Load from HDF
    f = h5py.File(mhw_mask_file, mode='r')
    mask = f['mask'][:,:,i0:i1+1]
    f.close()

    # Convert to IRIS
    tunit = Unit('days since 01-01-01 00:00:00', calendar='gregorian')
    times = np.arange(ts, te+1)
    time_coord = iris.coords.DimCoord(times, standard_name='time', units=tunit)

    # Space
    lat_coord, lon_coord = sst_utils.noaa_oi_coords(as_iris_coord=True)

    # Iris
    mcube = iris.cube.Cube(mask, var_name='Mask',
                           dim_coords_and_dims=[
                               (lat_coord, 0),
                               (lon_coord, 1),
                               (time_coord, 2), ])
    # Return
    return mcube
