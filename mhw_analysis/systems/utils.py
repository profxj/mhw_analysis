""" Utility routines related to MHW Systems"""
import numpy as np
import pandas
from datetime import date
from collections import Counter
from numba import njit, prange

from scipy.interpolate import interp1d

from oceanpy.sst import io as sst_io

from IPython import embed

def dict_to_pandas(sys_dict, add_latlon=False, add_date=True):
    """

    Parameters
    ----------
    sys_dict : dict
    add_latlon : bool, optional
        If True, load an NOAA OI and add lat, lon values to Table for convenience
    add_date : bool, optional
        If True, add dates and use as index

    Returns
    -------
    mhw_sys : pandas.DataFrame

    """
    mhw_sys = pandas.DataFrame()
    for key in sys_dict.keys():
        s = pandas.Series(sys_dict[key])
        mhw_sys[key] = s
    # Date?
    if add_date:
        date_max = [date.fromordinal(723546 + int(zcen)) for zcen in mhw_sys['zcen'].values]
        mhw_sys['date'] = date_max

    # Lon/Lat
    if add_latlon:
        # Could hard code this
        noaa = sst_io.load_noaa((1, 1, 2003))
        lat_coord = noaa.coord('latitude')
        lon_coord = noaa.coord('longitude')
        f_lat = interp1d(np.arange(lat_coord.points.size), lat_coord.points)
        f_lon = interp1d(np.arange(lon_coord.points.size), lon_coord.points)
        # Evaluate
        mhw_sys['lat'] = f_lat(np.minimum(mhw_sys['xcen'].values, 719.))
        mhw_sys['lon'] = f_lon(mhw_sys['ycen'].values)

    # Index
    mhw_sys = mhw_sys.set_index('Id')
    # Return
    return mhw_sys

'''
def tmp_max_area(mask):
    obj_id = np.unique(mask[mask > 0])
    areas = np.zeros_like(obj_id)
    for kk, id in enumerate(obj_id):
        idx = np.where(mask == id)
        areas[kk] = Counter(idx[2]).most_common(1)[0][1]
    return areas
'''

@njit(parallel=True)
def max_area(mask, obj_id, areas):

    for kk, id in enumerate(obj_id):
        idx = np.where(mask == id)
        unique = np.unique(idx[2])
        counts = 0
        for jj in unique:
            counts = max(counts, np.sum(idx[2]==jj))
        areas[kk] = counts


def prep_labels(mask, parent, NSpax, MinNSpax=0, verbose=False):
    # !..this is the number of individual connected components found in the cube:
    nlabels = np.max(mask)
    nobj=np.sum(parent[1:nlabels+1]==0)
    if verbose:
        print("NObj Extracted=",nobj)

    LabelToId = np.zeros(nlabels+1, dtype=np.int32) -1
    IdToLabel = np.zeros(nobj, dtype=np.int32)

    #!----- DETECTION (using NSpax) -------------
    # !..build auxiliary arrays and count detections
    ndet=0
    for i in range(1,nlabels+1):
        if parent[i] == 0:
            this_label = i
            this_NSpax = NSpax[this_label] # 0-indexing
            if this_NSpax > MinNSpax:
                IdToLabel[ndet] = this_label
                LabelToId[this_label] = ndet
                ndet = ndet + 1  # ! update ndet
    if verbose:
        print('Nobj Detected =', ndet)

    # Return
    return IdToLabel, LabelToId, ndet
