""" Module to deal with I/O of CEMS-LENS files"""
import os

import numpy as np

import datetime

import xarray


def climate_doy(doy, climate_file=None):
    """
    Return the NCEP Z500 for a given day of the year.

    Default is from the 30 years spanning 1983-2012 in the NOAA OI

    Parameters
    ----------
    doy : int

    Returns
    -------
    ncep_ds : xarray.Dataset
        Includes 'seasonalZ500' and 'threshZ500'

    """
    # File
    if climate_file is None:
        ncep_path = os.getenv("NCEP_DOE")
        if ncep_path is None:
            raise IOError("You muse set the NCEP_DOE environmental variable!")
        climate_file = os.path.join(ncep_path, 'NCEP-DOE_Z500_climate.nc')

    # Load
    ncep_climate = xarray.load_dataset(climate_file)

    # Select
    ncep_ds = ncep_climate.sel(day=doy)

    # Return
    return ncep_ds


def load_z500(dmy, ncep_path=None, end_dmy=None):
    """
    Load a Z500 cube from our NCEP nc file

    Parameters
    ----------
    dmy : tuple  (day, month, year)
        if end_date is not None, this is interpreted as the start date
    ncep_path : str, optional
        Path to NCEP-DOE data
    end_dmy : tuple, optional  (day, month, year)
        if provided, scan for a range of dates
        Note, this day is *not* inclusive

    Returns
    -------
    z500_dmy : xarray.DataArray
        lon,lat only if end_dmy is None
        else time,lon,lat

    """
    # File
    if ncep_path is None:
        ncep_path = os.getenv("NCEP_DOE")

    # Load cube
    ifile = os.path.join(ncep_path, 'NCEP-DOE_Z500.nc')
    ncep_xr = xarray.load_dataset(ifile)

    # Single Day
    day, month, year = dmy
    if end_dmy is None:
        ds = ncep_xr.sel(time=datetime.datetime(year=year, day=day, month=month))
    else:
        import pdb; pdb.set_trace()
        # NEED TO SET THIS UP
        time1 = iris.time.PartialDateTime(day=day, year=year, month=month)
        day2, month2, year2 = end_dmy
        time2 = iris.time.PartialDateTime(day=day2, year=year2, month=month2)
        constraint = iris.Constraint(time=lambda cell: time1 <= cell < time2)


    return ds
