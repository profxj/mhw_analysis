""" Module to deal with I/O of CEMS-LENS files"""

import os
import numpy as np

def load_z500(yyyymmdd, cubes=None, cems_path=None,
              dmy_name='current date (YYYYMMDD)',
              z500_name='Geopotential Z at 500 mbar pressure surface'):
    # Load cubes?
    if cubes is None:
        if cems_path is None:
            cems_path = os.getenv('CEMS_PATH')

    # Grab the day
    names = [cube.name() for cube in cubes]
    didx = names.index(dmy_name)
    day_cube = cubes[didx]
    dmyidx = np.where(yyyymmdd == day_cube.date[:])[0][0]

    # Grab the cube
    zidx = names.index(z500_name)
    z500_cube = cubes[zidx]
    z500_dmy = z500_cube[dmyidx]

    # Return
    return z500_dmy
