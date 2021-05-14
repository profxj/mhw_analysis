""" Module to analyze MHW systems

"""
import os
import numpy as np

from IPython import embed

import ctypes

# Mimics astropy convention
LIBRARY_PATH = os.path.dirname(__file__)
try:
    _systems = np.ctypeslib.load_library("_systems", LIBRARY_PATH)
except Exception:
    raise ImportError('Unable to load analysis C extension.  Try rebuilding mhw_analysis.')


spatial_systems_c = _systems.spatial_systems
spatial_systems_c.restype = None
spatial_systems_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # mask
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # shape
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # img
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # systems
                             ctypes.c_int,  # n_good
                             ctypes.c_int]  # tot_systems

days_in_systems_c = _systems.days_in_systems
days_in_systems_c.restype = None
days_in_systems_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # mask
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # shape
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # img
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]  # systems

days_in_systems_by_year_c = _systems.days_in_systems_by_year
days_in_systems_by_year_c.restype = None
days_in_systems_by_year_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # mask
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # shape
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # img
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # systems
                             np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]  # year array



def spatial_systems(mask, systems, max_Id):
    """
    Count up the number of times a given location
    is within the set of systems

    Args:
        mask: np.ndarray of int32
        systems:

    Returns:
        np.ndarray: spat_img (int32)

    """
    spat_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    spatial_systems_c(mask, np.array(mask.shape, dtype=np.int32), spat_img,
                  systems, len(systems), max_Id)

    return spat_img


def days_in_systems(mask, sys_flag):
    """
    Count up the number of times a given location
    is within the set of systems

    Args:
        mask: np.ndarray of int32
        sys_flag: np.ndarray of int64
            Labels whether the mask_ID is in the list of systems

    Returns:
        np.ndarray: spat_img (int32)
            Number of days that location was in a system

    """
    spat_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    days_in_systems_c(mask, np.array(mask.shape, dtype=np.int32), 
                      spat_img, sys_flag)

    return spat_img

def days_in_systems_by_year(mask, sys_flag, rel_year):
    """
    Count up the number of times a given location
    is within the set of systems, by year

    Args:
        mask: np.ndarray of int32
        sys_flag: np.ndarray of int32
            Labels whether the mask_ID is in the list of systems
        rel_year: np.ndarray of int32
            Converts day to relative year 

    Returns:
        np.ndarray: spat_img (int32)
            Number of days that location was in a system

    """
    nyear = rel_year.max() - rel_year.min() + 1
    spat_img = np.zeros((mask.shape[0], mask.shape[1], nyear), dtype=np.int32)
    shape = [mask.shape[0], mask.shape[1], mask.shape[2], nyear]
    days_in_systems_by_year_c(mask, np.array(shape, dtype=np.int32), 
                      spat_img, sys_flag, rel_year)

    return spat_img