""" Module to deal with I/O of CEMS-LENS files"""

import numpy as np
import os
from datetime import datetime
import iris

# could potentially make an I/O object to store cached things (ie names and cubes)

def load_z500(yyyymmdd, cubes, names, dmy_name='current date (YYYYMMDD)', 
                z500_name='Geopotential Z at 500 mbar pressure surface'):
    """
    Load a Z500 cube from a CEMS-LENS Cubes list

    Parameters
    ----------
    yyyymmdd : int
    cubes : iris.Cubes
    names : iris.Cubes (?), call load_z500_names prior to calling method
    dmy_name : str, optional
    z500_name : str, optional

    Returns
    -------
    z500_dmy : iris.Cube

    """
    # Load day
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

def load_z500_noaa(date, any_sst, cubes, names, path='../Z500', document='b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.Z500.18500101-20051231-043.nc'):
    # convert date to proper format
    ymd = date.year*10000 + date.month*100 + date.day

    # create filepath
    zfile = os.path.join(path, document)

    # get cube at ymd date
    z500_cube = load_z500(ymd, cubes, names)

    # get z500 regridded data
    z500_noaa = z500_cube.regrid(any_sst, iris.analysis.Linear())

    return z500_noaa

def load_cubes(path='../Z500', document='b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.Z500.18500101-20051231-043.nc'):
    return iris.load(os.path.join(path, document))

# get all names from cubes
def load_z500_names(cubes):
    return [cube.name() for cube in cubes]