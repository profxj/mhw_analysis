""" Script to find a MHW System"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Find a MHW System')
    parser.add_argument("date", type=str, help="Date of MHW System (YYYY-MM-DD)")
    parser.add_argument("lat", type=float, help="Latitude of MHW System ([-90,90] deg)")
    parser.add_argument("lon", type=float, help="Longitude of MHW System ([0, 360] deg)")
    parser.add_argument("--dataset", type=str, default='2019', help="MHW System set:  orig, vary")
    #parser.add_argument("plot_type", type=str, help="Plot type:  first_day")
    #parser.add_argument("-m", "--maskid", type=int, help="Mask Id")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import os
    import datetime

    import xarray

    from mhw_analysis.systems import io as mhw_sys_io
    from mhw_analysis.systems import defs

    vary = False
    if pargs.dataset == '2019':
        vary=False
        mask_file=os.path.join(os.getenv('MHW'), 'db', 
                               'MHWS_2019_mask.nc')
        mhw_sys_file=os.path.join(
            os.getenv('MHW'), 'db', 'MHWS_2019.csv')
    elif pargs.dataset == 'vary':
        vary=True
        embed(header='Need to add mask file; 43 of find_system script')
    else:
        raise IOError("Bad flavor!")

    # Load the systems
    mhws = mhw_sys_io.load_systems(vary=vary, mhw_sys_file=mhw_sys_file)

    # Create date
    year, month, day = [int(x) for x in pargs.date.split('-')]
    date = datetime.datetime(year, month, day)
    dt = datetime.timedelta(days=10)
    ds = date - dt
    de = date + dt

    # Location
    xlat = int((pargs.lat + 89.875) / 0.25)
    ylon = int((pargs.lon + 0.125) / 0.25)

    # Load mask
    mask = mhw_sys_io.load_mask_from_dates(
        (ds.year, ds.month, ds.day), 
        (de.year, de.month, de.day),
        mhw_mask_file=mask_file)

    # Grab IDs
    mhw_IDs = mask.data[xlat, ylon, :]

    # Find the best
    uni_IDs, uni_cnts = np.unique(mhw_IDs, return_counts=True)
    non_zero = uni_IDs > 0

    if not np.any(non_zero):
        print("No MHW Systems found")
        return
    
    imx = np.argmax(uni_cnts[non_zero])

    ID = uni_IDs[non_zero][imx]

    # Match and show
    i_mhws = np.where(mhws.mask_Id == ID)[0][0]
    print(mhws.iloc[i_mhws])

    # Characterize
    for iclss in defs.type_dict_km:
        if (mhws.iloc[i_mhws].NVox_km >= defs.type_dict_km[iclss][0]) & (
            (mhws.iloc[i_mhws].NVox_km < defs.type_dict_km[iclss][1])):
            clss = iclss
            break

    print("------------------------------------------------")
    print(f"This is a {clss} MHW System")
    print("------------------------------------------------")





