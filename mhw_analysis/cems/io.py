""" Module to deal with I/O of CEMS-LENS files"""

import numpy as np


def load_z500(yyyymmdd, cubes, dmy_name='current date (YYYYMMDD)',
              z500_name='Geopotential Z at 500 mbar pressure surface'):
    """
    Load a Z500 cube from a CEMS-LENS Cubes list

    Parameters
    ----------
    yyyymmdd : int
    cubes : iris.Cubes
    dmy_name : str, optional
    z500_name : str, optional

    Returns
    -------
    z500_dmy : iris.Cube

    """
    # Load day
    names = [cube.name() for cube in cubes]
    didx = names.index(dmy_name)
    day_cube = cubes[didx]

    # Checking
    if yyyymmdd > np.max(day_cube.data[:]):
        raise IOError("Your yyyymmdd exceeds the range of the models")

    # Grab
    dmyidx = np.where(yyyymmdd == day_cube.data[:])[0][0]

    # Grab the cube
    zidx = names.index(z500_name)
    z500_cube = cubes[zidx]
    z500_dmy = z500_cube[dmyidx]

    # Return
    return z500_dmy
