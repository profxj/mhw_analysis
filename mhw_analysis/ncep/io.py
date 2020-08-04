""" Module to deal with I/O of CEMS-LENS files"""
import os

import numpy as np

import iris


def load_z500(dmy, path=None, end_dmy=None):
    """
    Load a Z500 cube from our NCEP nc file

    Parameters
    ----------
    dmy : tuple  (day, month, year)
        if end_date is not None, this is interpreted as the start date
    path : str
        Path to NCEP-DOE data
    end_dmy : tuple, optional  (day, month, year)
        if provided, scan for a range of dates
        Note, this day is *not* inclusive

    Returns
    -------
    z500_dmy : iris.Cube

    """
    # File
    if path is None:
        path = os.getenv("NCEP_DOE")

    ifile = os.path.join(path, 'NCEP-DOE_Z500.nc')

    # Load cube
    cube = iris.load(ifile)

    # Single Day
    day, month, year = dmy
    if end_dmy is None:
        constraint = iris.Constraint(time=iris.time.PartialDateTime(
            day=day, year=year, month=month))
    else:
        time1 = iris.time.PartialDateTime(day=day, year=year, month=month)
        day2, month2, year2 = end_dmy
        time2 = iris.time.PartialDateTime(day=day2, year=year2, month=month2)
        constraint = iris.Constraint(time=lambda cell: time1 <= cell < time2)

    # Extract
    dmy_cube = cube.extract(constraint)

    return dmy_cube
