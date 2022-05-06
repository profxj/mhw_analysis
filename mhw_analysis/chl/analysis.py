import os
import xarray
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import pandas

from IPython import embed

def chl_for_mhws_date(mask_Id:int, date:str,
                         cut_mask_tuple:tuple=(-80.2, 90)):
    """_summary_

    Args:
        mask_Id (int): _description_
        date (str): YYYY-MM-DD

    Returns:
        mhws_chl (np.ndarray):
            CHL values in the MHWS
        mhws_base (np.ndarray):
            Baseline CHL values in the MHWS for the given DOY
        rc (np.ndarray):
            Array of relative change values
    """
    # Convert to datetime
    pd_datetime = pandas.to_datetime(date)

    # Open Chlorophyl dataset
    chl_data_path = os.path.join(os.getenv('CMEMS'), 'CHL')
    chl_file = os.path.join(chl_data_path,
                            f'chl_{pd_datetime.year}-{pd_datetime.month}.nc')
    chl_ds = xarray.open_dataset(chl_file)
    chl = chl_ds.sel(time=date).chl  # This has depth and time

    # Open baseline
    baseline_file = os.path.join(chl_data_path,
                                 'CHL_baseline_1993-2020.nc')
    bl=xarray.open_dataset(baseline_file)

    # Grab DOY
    doy = (pd_datetime - datetime(pd_datetime.year, 1, 1)).days
    baseline = bl.seasonalT.sel(doy=[doy])

    # Read mask
    mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_mask.nc')
    mask=xarray.open_dataset(mask_file,
                             engine='h5netcdf')
    mask_date = mask.sel(time=[date], lat=slice(cut_mask_tuple[0],
                                                cut_mask_tuple[1]))

    # Grab our MHWS
    mhws = mask_date.mask.data[...,0] == mask_Id
    bls_cut = baseline.data[0,...]
    chl_cut = chl.data[0,0,...]
    mhws_base = bls_cut[mhws]
    mhws_chl = chl_cut[mhws]

    # Relative change
    rc=((mhws_chl-mhws_base)/mhws_base)

    # Remove NaNs
    good_rc = np.isfinite(rc)
    rc = rc[good_rc]

    # Return
    return mhws_chl, mhws_base, rc

if __name__ == '__main__':
    # Test on the Blob!
    chl, base, rc = chl_for_mhws_date(531, '2016-12-31')
    embed(header='49 of analysis')
    m=np.mean(rc)
    print(m)

