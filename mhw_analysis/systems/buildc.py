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
    _build = np.ctypeslib.load_library("_build", LIBRARY_PATH)
except Exception:
    raise ImportError('Unable to load build C extension.  Try rebuilding mhw_analysis.')


#-----------------------------------------------------------------------
first_pass_c = _build.first_pass
first_pass_c.restype = None
first_pass_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

second_pass_c = _build.second_pass
second_pass_c.restype = None
second_pass_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

maxnlabels = 10000000

def first_pass(cube):
    # Init
    mask = np.zeros_like(cube, dtype=np.int32)
    # C
    #first_pass_c(cube, mask, cube.shape[0], cube.shape[1], cube.shape[2])
    parent = np.zeros(maxnlabels, dtype=np.int32)
    first_pass_c(cube, mask, np.array(cube.shape, dtype=np.int32), parent)
    # Return
    return mask, parent

def second_pass(mask, parent):
    """

    Args:
        mask: np.ndarray of int32
            Modified in place
        parent:

    Returns:
        np.ndarray: NSpax (int32)

    """
    NSpax = np.zeros(maxnlabels, dtype=np.int32)
    second_pass_c(mask, parent, np.array(mask.shape, dtype=np.int32), NSpax)

    return NSpax

def define_systems(cube, verbose=True, MinNSpax=0):
    """


    Args:
        cube:
        verbose:
        MinNSpax:

    Returns:

    """
    pass
