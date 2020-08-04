""" Module to deal with I/O of CEMS-LENS files"""
import os

import numpy as np

import iris


def load_z500(dmy, path=None):
    """
    Load a Z500 cube from our NCEP nc file

    Parameters
    ----------
    dmy : tuple  (day, month, year)
    path : str
        Path to NCEP-DOE data

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

    # Day
    day, month, year = dmy
    constraint = iris.Constraint(time=iris.time.PartialDateTime(
        day=day, year=year, month=month))
    dmy_cube = cube.extract(constraint)

    return dmy_cube
