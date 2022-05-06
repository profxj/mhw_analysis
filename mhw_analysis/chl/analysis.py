import os
import xarray
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime

from mhw_analysis.systems import io as mhw_sys_io

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
    doy = (pd_datetime - datetime(pd_datetime.year, 1, 1)).days + 1
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
    ac=(mhws_chl-mhws_base)[good_rc]

    # Return
    return mhws_chl, mhws_base, rc, ac

def mhws_time_series_rc(mask_Id, tstep:int=50,
                        mhw_sys_file:str=os.path.join(
                            os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
                        plot=False):

    # Grab the MHWS
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file)#, vary=vary)
    idx = np.where(mhw_sys.mask_Id == mask_Id)[0][0]
    mhws = mhw_sys.iloc[idx]

    # Loop on times
    ioff = 1
    itime = mhws.startdate + pandas.Timedelta(f'{ioff}days')
    mean_rcs = []
    mean_ac = []
    n_rcs = []
    offsets = []

    while (itime < mhws.enddate):
        str_date = f'{itime.year}-{itime.month}-{itime.day}'
        chl, base, rc, ac = chl_for_mhws_date(mask_Id, str_date)
        # More analysis
        # Save
        n_rcs.append(rc.size)
        mean_rcs.append(np.mean(rc))
        mean_ac.append(np.mean(ac))
        offsets.append(ioff)

        # Offset in time
        ioff += tstep
        itime = mhws.startdate + pandas.Timedelta(f'{ioff}days')
        print(itime)

    # Plot
    if plot:
        plt.clf()
        gs = gridspec.GridSpec(3,1)

        # Mean rc
        ax_mean = plt.subplot(gs[0])
        ax_mean.plot(offsets, mean_rcs)
        #
        ax_mean.axhline(0., color='k', ls='--')
        ax_mean.set_ylabel('Mean rc')

        # Mean ac
        ax_meana = plt.subplot(gs[1])
        ax_meana.plot(offsets, mean_ac)
        #
        #ax_meana.axhline(0., color='k', ls='--')
        ax_meana.set_ylabel('Mean ac')

        # Number
        ax_n = plt.subplot(gs[-1])
        ax_n.plot(offsets, n_rcs, color='g')
        ax_n.set_xlabel('Offset from start (days)')
        ax_n.set_ylabel('Number of cells')
        plt.show()

    embed(header='136 of analysis')


def mhws_histogram():
    pass


if __name__ == '__main__':

    ## ############ 
    # Test time series 
    ## ############ 

    # #1
    #mhws_time_series_rc(58877, plot=True)

    # #3
    mhws_time_series_rc(36016, plot=True)

    # Blob
    #mhws_time_series_rc(531, plot=True)

    '''
    # Test rc on the Blob!
    chl, base, rc, ac = chl_for_mhws_date(531, '2016-12-31')
    embed(header='49 of analysis')
    m=np.mean(rc)
    print(m)
    '''

