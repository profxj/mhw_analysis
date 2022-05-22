""" Methods for analyzing MHWS """
import sys, os
import numpy as np
import datetime

import xarray

from mhw_analysis.systems import io as mhw_sys_io
from mhw_analysis.systems import analysisc

from IPython import embed 

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import defs

def count_days_by_year(mhw_sys_file, mask_file,
                       outfile='extreme_days_by_year.nc',
                       use_km=True,
                       mhw_type=defs.classc, debug=False):
    # Load systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file)#, vary=vary)

    if use_km:
        type_dict = defs.type_dict_km
        NVox = mhw_sys.NVox_km
    else:
        type_dict = defs.type_dict
        NVox = mhw_sys.NVox

    # Prep
    if debug:
        print("Loading only a subset of the mask for debuggin")
        mask = mhw_sys_io.maskcube_from_slice(0, 4000, vary=True)
    else:
        mask = mhw_sys_io.load_full_mask(mhw_mask_file=mask_file)
    
    # Generate year array
    d0=datetime.date(1982,1,1)
    t0 = d0.toordinal()
    t = t0 + np.arange(mask.shape[2])
    dates = [datetime.datetime.fromordinal(it) for it in t]
    year = np.array([idate.year for idate in dates])
    rel_year = year-1982

    # Systems
    print("Cut down systems..")
    sys_flag = np.zeros(mhw_sys.mask_Id.max()+1, dtype=int)
    NVox_mxn = type_dict[mhw_type]
    cut = (NVox >= NVox_mxn[0]) & (NVox <= NVox_mxn[1])
    cut_sys = mhw_sys[cut]
    sys_flag[cut_sys.mask_Id] = 1

    # Run it
    print("Counting the days...")
    days_by_year = analysisc.days_in_systems_by_year(
                    mask.data, 
                    sys_flag.astype(np.int32),
                    rel_year.astype(np.int32))
    
    # Write
    # Grab coords
    noaa_path = os.getenv("NOAA_OI")
    climate_cube_file = os.path.join(noaa_path, 'NOAA_OI_climate_1983-2012.nc')
    clim = xarray.open_dataset(climate_cube_file)

    time_coord = xarray.IndexVariable('year', 1982 + np.arange(days_by_year.shape[2]).astype(int))
    da = xarray.DataArray(days_by_year, 
                          coords=[clim.lat, clim.lon, time_coord])
    ds = xarray.Dataset({'ndays': da})
    ds.to_netcdf(outfile, engine='h5netcdf')
    print("Wrote: {}".format(outfile))

def main(flg_main):
    if flg_main == 'all':
        flg_main = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_main = int(flg_main)

    # Hobday
    if flg_main & (2 ** 0):
        outfile='extreme_dy_by_yr_defaults.nc'
        mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_defaults.csv')
        mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_defaults_mask.nc')
        count_days_by_year(mhw_sys_file, mask_file, outfile=outfile)
    
    # 2019, detrend
    if flg_main & (2 ** 1):
        outfile='extreme_dy_by_yr_2019_local_detrend.nc'
        mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_local.csv')
        mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_local_mask.nc')
        count_days_by_year(mhw_sys_file, mask_file, outfile=outfile)

    # 2019
    if flg_main & (2 ** 2):
        mhw_sys_file=os.path.join(os.getenv('MHW'), 
                                  'db', 'MHWS_2019.csv')
        mask_file=os.path.join(os.getenv('MHW'), 
                               'db', 'MHWS_2019_mask.nc')
        # Severe
        outfile=f'{defs.classc}_km_dy_by_yr_2019.nc'
        count_days_by_year(mhw_sys_file, mask_file, 
                           outfile=outfile, use_km=True)

        outfile=f'{defs.classb}_km_dy_by_yr_2019.nc'
        count_days_by_year(mhw_sys_file, mask_file, 
                           outfile=outfile, 
                           mhw_type=defs.classb, use_km=True)
        outfile=f'{defs.classa}_km_dy_by_yr_2019.nc'
        count_days_by_year(mhw_sys_file, mask_file, 
                           outfile=outfile, 
                           mhw_type=defs.classa, use_km=True)

# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg_main = 0
        #flg_main += 2 ** 0  # Defaults
        #flg_main += 2 ** 1  # Days by year, 2019 detrend local
        flg_main += 2 ** 2  # Days by year, 2019 
    else:
        flg_main = sys.argv[1]

    main(flg_main)