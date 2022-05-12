import os
import xarray
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime

from mhw_analysis.systems import io as mhw_sys_io

import pandas

from IPython import embed
from dateutil.relativedelta import relativedelta


def chl_for_mhws_date(mask_Id:int, date:str,
                         cut_mask_tuple:tuple=(-80.2, 90),year_offset=0):
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
                            f'chl_{pd_datetime.year+year_offset}-{pd_datetime.month}.nc')
    chl_ds = xarray.open_dataset(chl_file)
    other_date= f'{pd_datetime.year+year_offset}-{pd_datetime.month}-{pd_datetime.day}'
    #embed(header="line 41")
    chl = chl_ds.sel(time=other_date).chl  # This has depth and time

    # Open baseline
    baseline_file = os.path.join(chl_data_path,
                                 'CHL_baseline_1993-2020.nc')
    bl=xarray.open_dataset(baseline_file)

    # Grab DOY
    #embed(header="line 53")
    doy = (pd_datetime+relativedelta(years=year_offset) - datetime(pd_datetime.year+year_offset, 1, 1)).days + 1
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
    #y=3650 
    #sd=mhws.startdate - pandas.Timedelta(f'{y}days')
    #ed=mhws.enddate - pandas.Timedelta(f'{y}days')
    ioff = 1
    itime = mhws.startdate + pandas.Timedelta(f'{ioff}days')
    mean_rcs_control = []
    mean_ac_control = []
    n_rcs_control = []
    n_acs_control=[]
    offsets_control = []
    rc_std_control=[]
    ac_std_control=[]
    final_rcc_mean=0
    final_acc_mean=0
    final_rcc_n=0
    final_acc_n=0
    final_rcc_std=0
    final_acc_std=0
    
    

    mean_rcs= []
    mean_ac= []
    n_rcs= []
    n_acs=[]
    offsets = []
    rc_std=[]
    ac_std=[]
    final_rc_mean=0
    final_ac_mean=0
    final_rc_n=0
    final_ac_n=0
    final_rc_std=0
    final_ac_std=0

    z_score_rc=0
    z_score_ac=0
    

    
    
    while (itime < mhws.enddate):
        str_date = f'{itime.year}-{itime.month}-{itime.day}'
        #control
        chl_control, base_control, rc_control, ac_control = chl_for_mhws_date(mask_Id, str_date,
            year_offset=-10)
        # More analysis
        # Save

        n_rcs_control.append(rc_control.size)
        n_acs_control.append(ac_control.size)
        mean_rcs_control.append(np.mean(rc_control))
        mean_ac_control.append(np.mean(ac_control))
        offsets_control.append(ioff)
        rc_std_control.append(np.std(rc_control))
        ac_std_control.append(np.std(ac_control))
       


        #MHWS
        chl, base, rc, ac = chl_for_mhws_date(mask_Id, str_date)
        # More analysis
        # Save
        n_rcs.append(rc.size)
        n_acs.append(ac.size)
        mean_rcs.append(np.mean(rc))
        mean_ac.append(np.mean(ac))
        offsets.append(ioff)
        rc_std.append(np.std(rc))
        ac_std.append(np.std(ac))
        
        #Analysis
        #z_score_rc=((np.array(mean_rcs)-np.array(mean_rcs_control))/(np.sqrt(((np.array(np.square(rc_std)))/np.array(n_rcs))+((np.array(np.square(rc_std_control)))/np.array(n_rcs_control)))))
        #z_score_ac=((np.array(mean_ac)-np.array(mean_ac_control))/(np.sqrt(((np.array(np.square(ac_std)))/np.array(n_acs))+((np.array(np.square(ac_std_control)))/np.array(n_acs_control)))))
        #print(z_score_rc)
        #print(z_score_ac)
        # Offset in time
        ioff += tstep
        itime = mhws.startdate + pandas.Timedelta(f'{ioff}days')
        print(itime)
        

    final_rcc_mean=np.mean(mean_rcs_control)
    final_acc_mean=np.mean(mean_ac_control)
    final_rcc_n=np.size(mean_rcs_control)
    final_acc_n=np.size(mean_ac_control)
    final_rcc_std=np.std(mean_rcs_control)
    final_acc_std=np.std(mean_ac_control)

    final_rc_mean=np.mean(mean_rcs)
    final_ac_mean=np.mean(mean_ac)
    final_rc_n=np.size(mean_rcs)
    final_ac_n=np.size(mean_ac)
    final_rc_std=np.std(mean_rcs)
    final_ac_std=np.std(mean_ac)
   

    z2_score_rc=((final_rc_mean-final_rcc_mean)/((np.sqrt((np.square(final_rc_std))/final_rc_n))+((np.square(final_rcc_std))/final_rcc_n)))
    z2_score_ac=((final_ac_mean-final_acc_mean)/((np.sqrt((np.square(final_ac_std))/final_ac_n))+((np.square(final_acc_std))/final_acc_n)))
    z1_score_rc=((final_rc_mean-final_rcc_mean)/(np.sqrt((np.square(final_rc_std))/final_rc_n)))
    z1_score_ac=((final_ac_mean-final_acc_mean)/(np.sqrt((np.square(final_ac_std))/final_ac_n)))

    print(z2_score_rc)
    print(z2_score_ac)
    print(z1_score_rc)
    print(z1_score_ac)

    # Plot control
    if plot:
        plt.clf()
        gs_control = gridspec.GridSpec(3,1)

        # Mean rc
        ax_mean_control = plt.subplot(gs_control[0])
        ax_mean_control.plot(offsets_control, mean_rcs_control)
        #
        ax_mean_control.axhline(0., color='k', ls='--')
        ax_mean_control.set_ylabel('Control Mean rc')

        # Mean ac
        ax_meana_control = plt.subplot(gs_control[1])
        ax_meana_control.plot(offsets_control, mean_ac_control)
        #
        #ax_meana.axhline(0., color='k', ls='--')
        ax_meana_control.set_ylabel('Control Mean ac')

        # Number
        ax_n_control = plt.subplot(gs_control[-1])
        ax_n_control.plot(offsets_control, n_rcs_control, color='g')
        ax_n_control.set_xlabel('Offset from start (days)')
        ax_n_control.set_ylabel('Control Number of cells')
        plt.show()

        #Plot MHWS
    
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

    

    #embed(header='136 of analysis')


def mhws_histogram():
    pass






if __name__ == '__main__':

    ## ############ 
    # Test time series 
    ## ############ 

    # #1
    #mhws_time_series_rc(58877, plot=True)

    # #3
    mhws_time_series_rc(531, plot=True)
   
    

    #mhws_time_series_rc_control(531, plot=True)

    # Blob
    #mhws_time_series_rc(531, plot=True)

    '''
    # Test rc on the Blob!
    chl, base, rc, ac = chl_for_mhws_date(531, '2016-12-31')
    embed(header='49 of analysis')
    m=np.mean(rc)
    print(m)
    '''

