""" I/O for MHW Systems"""
import os

import pandas


def load_systems(mhw_sys_file=None):
    if mhw_sys_file is None:
        mhw_sys_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_systems.hdf')
    # Read
    mhw_sys = pandas.read_hdf(mhw_sys_file)
    # Return
    return mhw_sys