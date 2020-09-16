""" I/O for MHW Systems"""
import os
import datetime
import numpy as np

import pandas
import h5py

import xarray

#from cf_units import Unit
#import iris

from oceanpy.sst import utils as sst_utils

from IPython import embed

def grab_mhw_sys_file(vary=False):
    if vary:
        mhw_sys_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_systems_vary.csv.gz')
    else:
        mhw_sys_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_systems.hdf')
    return mhw_sys_file

def grab_mhwsys_mask_file(vary=False):
    if vary:
        mhwsys_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf')
    else:
        mhwsys_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask.hdf')
    return mhwsys_mask_file


def load_systems(mhw_sys_file=None, vary=False):
    """
    Load up the MHW Systems into a pandas table

    Args:
        mhw_sys_file:
        vary (bool, optional):
            If True, load up the MHW Systems with varying climate threshold

    Returns:
        pandas.DataFrame:

    """
    if mhw_sys_file is None:
        mhw_sys_file = grab_mhw_sys_file(vary=vary)
    # Read
    print("Loading systems from {}".format(mhw_sys_file))
    mhw_sys = pandas.read_csv(mhw_sys_file)
    # Return
    return mhw_sys


def load_mask_from_dates(ymd_start, ymd_end,
                         mhw_mask_file=None, mask_start=(1982,1,1),
                         vary=False):
    """
    Load a portion of a MHWS mask for a range of dates

    Args:
        ymd_start (tuple):
            Starting date;  inclusive
        ymd_end (tuple):
            End date;  inclusive
        mhw_mask_file:
        mask_start:

    Returns:
        xarray.DataArray:

    """
    # Convert ymd to indices
    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()

    ts = datetime.date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal()
    te = datetime.date(ymd_end[0], ymd_end[1], ymd_end[2]).toordinal()

    i0 = ts-t0
    i1 = te-t0  # Note: maskcube_from_slice makes it inclusive

    # Load + return
    return maskcube_from_slice(i0, i1, mhw_mask_file=mhw_mask_file, mask_start=mask_start,
                               vary=vary)


def load_mask_from_system(mhw_system,
                          mhw_mask_file=None, mask_start = (1982, 1, 1), vary=False):
    # Load + return
    print("Loading mask from system: \n {}".format(mhw_system))
    return maskcube_from_slice(mhw_system.zboxmin, mhw_system.zboxmax,
                               mhw_mask_file=mhw_mask_file, mask_start=mask_start,
                               vary=vary)

def load_full_mask(mhw_mask_file=None, vary=False):
    return maskcube_from_slice(0, -1, mhw_mask_file=mhw_mask_file, vary=vary)


def maskcube_from_slice(i0,i1,
                        mhw_mask_file=None, mask_start=(1982, 1, 1),
                        vary=False):
    """

    Parameters
    ----------
    i0 : int
    i1 : int
        If i1=-1, grab the data to the end
    mhw_mask_file : str, optional
    mask_start : tuple, optional

    Returns
    -------
    da : xarray.DataArray

    """
    if mhw_mask_file is None:
        mhw_mask_file = grab_mhwsys_mask_file(vary=vary)

    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()
    ts = datetime.datetime.fromordinal(t0 + i0)

    # Load from HDF
    print("Loading mask from {}".format(mhw_mask_file))
    f = h5py.File(mhw_mask_file, mode='r')
    if i1 == -1:
        mask = f['mask'][:,:,i0:]
    else:
        mask = f['mask'][:,:,i0:i1+1]
    f.close()

    # Convert to xarray DataSet
    # Space
    lat_coord, lon_coord = sst_utils.noaa_oi_coords()

    # Xarray
    times = pandas.date_range(start=ts, periods=mask.shape[2])
    da = xarray.DataArray(mask, coords=[lat_coord, lon_coord, times],
                          dims=['lat', 'lon', 'time'])

    # Return
    return da
