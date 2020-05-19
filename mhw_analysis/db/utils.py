import iris
import numpy as np

def load_noaa_sst(sst_files):
    """

    Args:
        sst_files (list):

    Returns:
        list:  iris Cube objects

    """
    all_sst = []
    for ifile in sst_files:
        print(ifile)  # For progress
        cubes = iris.load(ifile)
        sst = cubes[0]
        # Get out of lazy
        _ = sst.data
        # Append
        all_sst.append(sst)
    #
    return all_sst

def grab_t(sst_list):
    allts = []
    for sst in sst_list:
        allts += (sst.coord('time').points + 657072).astype(int).tolist()  # 1880?
    return np.array(allts)

def grab_T(sst_list, i, j):
    allTs = []
    for sst in sst_list:
        allTs += [sst.data[:,i,j]]
    return np.ma.concatenate(allTs)
