
import os, sys
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units

import pandas

import datetime
from datetime import date

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import defs

def cut_by_type(mhw_sys):
    """


    Args:
        mhw_sys (pandas.DataFrame):

    Returns:

    """
    small = mhw_sys.NVox < defs.type_dict['random'][1]
    normal = (mhw_sys.NVox >= defs.type_dict['normal'][0]) & (
            mhw_sys.NVox < defs.type_dict['normal'][1])
    extreme = (mhw_sys.NVox >= defs.type_dict['extreme'][0])

    # Return
    return small, normal, extreme


def Nvox_by_year(mhw_sys):
    """

    Args:
        mhw_sys (pandas.DataFrame):

    Returns:

    """

    # Do the stats
    years = 1983 + np.arange(37)

    small_Nvox = np.zeros_like(years)
    int_Nvox = np.zeros_like(years)
    ex_Nvox = np.zeros_like(years)

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
        nvox_in_year = mhw_sys.NVox * ndays.to_numpy().astype(float) / mhw_sys.duration.to_numpy().astype(float)

        # Cut em up
        small_Nvox[jj] = np.sum(nvox_in_year[in_year & random])
        int_Nvox[jj] = np.sum(nvox_in_year[in_year & normal])
        ex_Nvox[jj] = np.sum(nvox_in_year[in_year & extreme])

    # Fill me up
    df = pandas.DataFrame(
        dict(random=small_Nvox, 
             normal=int_Nvox, 
             extreme=ex_Nvox),
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


