
import os, sys
from matplotlib import use
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units

import pandas

import datetime
from datetime import date

from IPython import embed

from mhw_analysis.systems import defs

# Local
sys.path.append(os.path.abspath("../Analysis/py"))

def cut_by_type(mhw_sys):
    """


    Args:
        mhw_sys (pandas.DataFrame):

    Returns:

    """
    small = mhw_sys.NVox < defs.type_dict[defs.classa][1]
    normal = (mhw_sys.NVox >= defs.type_dict[defs.classb][0]) & (
            mhw_sys.NVox < defs.type_dict[defs.classb][1])
    extreme = (mhw_sys.NVox >= defs.type_dict[defs.classc][0])

    # Return
    return small, normal, extreme


def cell_area_by_lat(R_Earth=6371.,cell_deg = 0.25):
    """Cell area in km^2 as a function of latitutde

    Args:
        nlat ([type]): [description]
        R_Earth ([type], optional): [description]. Defaults to 6371..
        cell_deg (float, optional): [description]. Defaults to 0.25.

    Returns:
        [type]: [description]
    """
    nlat = int(180./cell_deg)

    cell_km = cell_deg * 2 * np.pi * R_Earth / 360.
    lat = -89.875 + cell_deg*np.arange(nlat)
    cell_lat = cell_km * cell_km * np.cos(np.pi * lat / 180.)
    return cell_lat
    
def Nvox_by_year(mhw_sys, use_km=True):
    """

    Args:
        mhw_sys (pandas.DataFrame):

    Returns:
        pandas.DataFrame: 

    """
    if use_km:
        attr = 'NVox_km'
        dt = float
    else:
        attr = 'NVox'
        dt = int

    # Do the stats
    years = 1983 + np.arange(37)

    small_Nvox = np.zeros_like(years, dtype=dt)
    int_Nvox = np.zeros_like(years, dtype=dt)
    ex_Nvox = np.zeros_like(years, dtype=dt)

    # Types
    random, normal, extreme = cut_by_type(mhw_sys)
    #
    for jj, year in enumerate(years):

        # Days in that year
        day_beg = np.maximum(mhw_sys.startdate, np.datetime64(datetime.datetime(year,1,1)))
        day_end = np.minimum(mhw_sys.enddate,
                             np.datetime64(datetime.datetime(year+1,1,1)))  # Should subtract a day
        ndays = day_end-day_beg
        in_year = ndays > datetime.timedelta(days=0)

        # Fraction 
        #  This is approximate for NVox_km
        nvox_in_year = mhw_sys[attr] * ndays.to_numpy().astype(float) / mhw_sys.duration.to_numpy().astype(float)

        # Cut em up
        small_Nvox[jj] = np.sum(nvox_in_year[in_year & random])
        int_Nvox[jj] = np.sum(nvox_in_year[in_year & normal])
        ex_Nvox[jj] = np.sum(nvox_in_year[in_year & extreme])

    # Fill me up
    df = pandas.DataFrame(
        {defs.classa: small_Nvox, 
             defs.classb: int_Nvox, 
             defs.classc: ex_Nvox},
        index=years)

    # Return
    return df


def random_sys(mhw_sys, years, dyear, type=None,
               min_dur=None, min_area=None,
               verbose=True, seed=12345):

    # Cuts
    cuts = np.ones(len(mhw_sys), dtype='bool')

    # Type?
    if type is not None:
        cut_type = (mhw_sys.NVox >= defs.type_dict[type][0]) & (
                mhw_sys.NVox < defs.type_dict[type][1])
        cuts &= cut_type

    # Minimum duration?
    if min_dur is not None:
        cut_dur = mhw_sys.duration > min_dur
        cuts &= cut_dur

    # Minimum area?
    if min_area is not None:
        cut_area = mhw_sys.max_area > min_area
        cuts &= cut_area

    # Cut
    cut_sys = mhw_sys[cuts]

    coords = SkyCoord(b=cut_sys.lat * units.deg, l=cut_sys.lon * units.deg, frame='galactic')

    rstate = np.random.RandomState(seed=seed)

    # Loop time
    used_coords = None
    for_gallery = []
    for year in years:
        # Cut on date
        t0 = datetime.datetime(year, 1, 1)
        t1 = datetime.datetime(year + dyear, 1, 1)
        in_time = np.where((cut_sys.datetime >= t0) & (cut_sys.datetime < t1))[0]
        if verbose:
            print('Year {}:, n_options={}'.format(year, len(in_time)))
        # Grab one
        if used_coords is not None:
            # Ugly loop
            all_seps = np.zeros((len(used_coords), len(in_time)))
            for kk, ucoord in enumerate(used_coords):
                seps = ucoord.separation(coords[in_time])
                all_seps[kk, :] = seps.to('deg').value
            # Minimum for each
            min_seps = np.min(all_seps, axis=0)
            best = np.argmax(min_seps)
            for_gallery.append(in_time[best])
            used_coords = coords[np.array(for_gallery)]
        else:
            # Take a random one
            rani = rstate.randint(low=0, high=len(in_time), size=1)[0]
            for_gallery.append(in_time[rani])
            used_coords = coords[np.array(for_gallery)]

    # Return table of random choices
    return cut_sys.iloc[for_gallery]


