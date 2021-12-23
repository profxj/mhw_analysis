""" Module to generate MHW systems

Some of the code that follows is from the routine “Extract.f90” in CubEx by S. Cantalupo.
That’s the core of the code. The inputs are the Datacube, another cube for the
flagging criteria (the Variance for astronomical cubes) and the output is a 3D segmentation map
or mask (called “Mask” or “mask”). The routine WriteCheckCubes.f90 is where this 3D Mask is used
for various sort of outputs.

Please note that for any astronomical purposes, the LICENCE in the folder applies
(this includes any modification of the original code).   Here it is:

Copyright: Sebastiano Cantalupo 2019.

CubEx (and associated software in the folder "Tools" and "Scripts") non-public software licence notes.

This is a non-public software:

1) you cannot distribute any part of this software without the authorization by the copyright holder.

2) you can modify any part of the software, however, any modifications do not change the copyright 
    and distribution rights stated in the point 1 above.

3) publications using results obtained with any part of this software in the original of modified form
    as stated in point 2, should include the copyright owner in the author list unless otherwise stated by
    the copyright owner.  

4) downloading, opening, installing and/or using the package is equivalent to accepting points 1, 2 and 3 above.
"""
import os
import numpy as np

import ctypes


from IPython import embed

# Mimics astropy convention
LIBRARY_PATH = os.path.dirname(__file__)
try:
    _systems = np.ctypeslib.load_library("_systems", LIBRARY_PATH)
except Exception:
    raise ImportError('Unable to load systems C extension.  Try rebuilding mhw_analysis.')


def is_loaded(lib):
    libp = os.path.abspath(lib)
    ret = os.system("lsof -p %d | grep %s > /dev/null" % (os.getpid(), libp))
    return (ret == 0)

def reload_lib(lib):
    handle = lib._handle
    name = lib._name
    del lib
    while is_loaded(name):   
        libdl = ctypes.CDLL("libdl.so")
        libdl.dlclose(handle)
    return ctypes.cdll.LoadLibrary(name)


#-----------------------------------------------------------------------
first_pass_c = _systems.first_pass
first_pass_c.restype = None
first_pass_c.argtypes = [
    #np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]


maxnlabels = 100000000

def first_pass(cube:np.ndarray):
    """ Perform the first step in 
    MHWS construction

    Args:
        cube (np.ndarray): lat,lon,time cube of MHWEs

    Returns:
        tuple: np.ndarray (mask), np.ndarray (parent), np.ndarray (category)
    """
    # Init
    mask = np.zeros_like(cube, dtype=np.int32)
    # C
    #first_pass_c(cube, mask, cube.shape[0], cube.shape[1], cube.shape[2])
    parent = np.zeros(maxnlabels, dtype=np.int32)
    category = np.zeros(maxnlabels, dtype=np.int32)
    print("Entering first_pass_c")
    first_pass_c(cube, mask, np.array(cube.shape, dtype=np.int32), parent, category)
    # Return
    return mask, parent, category

second_pass_c = _systems.second_pass
second_pass_c.restype = None
second_pass_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

def second_pass(mask:np.ndarray, parent:np.ndarray, category:np.ndarray):
    """
    Perform the second step in MHWS construction

    Args:
        mask (np.ndarray, int32):
            Modified in place
        parent (np.ndarray):
        category (np.ndarray):

    Returns:
        np.ndarray: NVox (int32) -- Volume of each MHWS

    """
    NVox = np.zeros(maxnlabels, dtype=np.int64)
    second_pass_c(mask, parent, np.array(mask.shape, dtype=np.int32), NVox, category)

    return NVox


final_pass_c = _systems.final_pass
final_pass_c.restype = None
final_pass_c.argtypes = [ctypes.c_int,
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"), # NVox
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # category
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # Label
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # category
                         ]


def final_pass(mask:np.ndarray, NVox:np.ndarray, ndet:int, 
               IdToLabel:np.ndarray, 
               LabelToId:np.ndarray, category:np.ndarray):
    """ Last step in MHWS construction

    Also measures some basic metrics on the MHWS

    Args:
        mask (np.ndarray): [description]
        NVox (np.ndarray): [description]
        ndet (int): [description]
        IdToLabel (np.ndarray): [description]
        LabelToId (np.ndarray): [description]
        category (np.ndarray): [description]

    Returns:
        dict: contains metrics on MHWS
    """
    # Objects
    obj_dict = dict(Id=np.zeros(ndet, dtype=np.int32),
                    NVox=np.zeros(ndet, dtype=np.int64),
                    category=np.zeros(ndet, dtype=np.int32), # Assoc=[0]*ndet,
                    mask_Id=np.zeros(ndet, dtype=np.int32),
                    max_area=np.zeros(ndet, dtype=np.int32),
                    xcen=np.zeros(ndet, dtype=np.float32), xboxmin=np.ones(ndet, dtype=np.int32)*100000, xboxmax=np.ones(ndet, dtype=np.int32)*-1,
                    ycen=np.zeros(ndet, dtype=np.float32), yboxmin=np.ones(ndet, dtype=np.int32)*100000, yboxmax=np.ones(ndet, dtype=np.int32)*-1,
                    zcen=np.zeros(ndet, dtype=np.float32), zboxmin=np.ones(ndet, dtype=np.int32)*100000, zboxmax=np.ones(ndet, dtype=np.int32)*-1)
    # Init
    for ii in range(ndet):
        obj_dict['Id'][ii] = ii+1
        obj_dict['NVox'][ii] = NVox[IdToLabel[ii]]
        obj_dict['mask_Id'][ii] = IdToLabel[ii]

    final_pass_c(ndet, mask, np.array(mask.shape, dtype=np.int32),
                 obj_dict['xcen'], obj_dict['ycen'], obj_dict['zcen'],
                 obj_dict['xboxmin'], obj_dict['xboxmax'],
                 obj_dict['yboxmin'], obj_dict['yboxmax'],
                 obj_dict['zboxmin'], obj_dict['zboxmax'],
                 obj_dict['NVox'], obj_dict['category'],
                 LabelToId, category,
                 )
    return obj_dict

max_areas_c = _systems.max_areas
max_areas_c.restype = None
max_areas_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ctypes.c_int,
                        np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                        ctypes.c_float,
                        ]


def max_areas(mask:np.ndarray, obj_dict:dict, cell_deg=0.25):
    """
    Calculate the maximum area of each MHWS

    Done as an afterburner...

    Args:
        mask (np.ndarray): [description]
        obj_dict (dict): [description]
        cell_deg (float): Cell size in deg
    """
    max_label = np.max(obj_dict['mask_Id'])
    areas = np.zeros(max_label+1, dtype=np.int32)
    areas_km2 = np.zeros(max_label+1, dtype=np.float32)

    # Run
    max_areas_c(mask, areas, areas_km2, max_label, 
                np.array(mask.shape, dtype=np.int32),
                cell_deg)
    # Fill
    for kk, label in enumerate(obj_dict['mask_Id']):
        obj_dict['max_area'][kk] = areas[label]

