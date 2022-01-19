""" Code to get Bruno up and running
"""
import numpy as np
import datetime
import os

import h5py
import pandas
import xarray

from oceanpy.sst import io as sst_io
from mhw_analysis.systems import io as mhw_sys_io
from IPython import embed

def grab_doy(dt):
    day_of_year = (dt - datetime.datetime(dt.year, 1, 1)).days + 1
    # Leap year?
    if (dt.year % 4) != 0:
        if day_of_year <= 59:
            pass
        else:
            day_of_year += 1
    # Return
    return day_of_year

def build_mhws_grid(outfile, mhw_sys_file=os.path.join(
                            os.getenv('MHW'), 'db', 'MHWS_2019.csv'),
                    vary=False,
                     #mask_Id=1468585,  -- Hobday
                     #mask_Id=1475052, # 2019, local
                     mask_Id=1458524, # 2019
                     find=False):
    #1489500):

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    if find:
        gd_dur = mhw_sys.duration > pandas.Timedelta(days=120)
        gd_lat = (mhw_sys.lat > 0.) & (mhw_sys.lat < 50.)
        gd_lon = (mhw_sys.lon > 190.) & (mhw_sys.lon < 250.)
        all_gd = gd_dur & gd_lat & gd_lon
        mhw_sys[all_gd]
        embed(header='22 of build')

    # Find a sys
    idx = np.where(mhw_sys.mask_Id == mask_Id)[0][0]
    isys = mhw_sys.iloc[idx]
    sys_startdate = isys.datetime - datetime.timedelta(days=int(isys.zcen)-int(isys.zboxmin))

    # Grab the mask
    mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_mask.nc')
    mask_da = mhw_sys_io.load_mask_from_system(isys, vary=vary,
                                               mhw_mask_file=mask_file)
    
    # Patch
    fov = 80.  # deg
    lat_min = isys.lat - fov/2.
    lat_max = isys.lat + fov/2.
    lon_min = isys.lon - fov/2.
    lon_max = isys.lon + fov/2.

    # T thresh
    climate_file='/data/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_detrend_local_climate_1983-2019.nc'
    ncep_climate = xarray.open_dataset(climate_file)

    toffs = np.arange(isys.duration.days)

    # Mayavi
    final_grid = mask_da.sel(lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max)).data[:].astype(float)

    for ss, toff in enumerate(toffs):
        print('ss: {}'.format(ss))

        # Grab SST
        if ss == 0:
            off_date = sys_startdate 
        else:
            off_date = sys_startdate + pandas.Timedelta(days=toff)
        sst = sst_io.load_noaa((off_date.day, 
                                off_date.month, 
                                off_date.year),
                               climate_file=climate_file)
        # Slice
        sst_slice = sst.sel(lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max))
        sst_img = sst_slice.data[:]

        # Threshold
        doy = grab_doy(off_date)
        Tdoy = ncep_climate.threshT.sel(doy=doy, 
                                        lon=slice(lon_min, lon_max), 
                                        lat=slice(lat_min, lat_max))
        Tdoy_img = Tdoy.data[:]

        # Slice the mask too
        mask_cut = mask_da.sel(time=off_date, 
                               lon=slice(lon_min, lon_max), 
                               lat=slice(lat_min, lat_max))
        mask = mask_cut.data[:]

        region = mask == mask_Id

        # Tdiff
        Tdiff = sst_img-Tdoy_img
        Tdiff[~region] = 0.

        # Save
        final_grid[:,:,ss] = Tdiff 

    # save
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('grid', data=final_grid)
    print('Wrote {:s}'.format(outfile))

# Command line execution
if __name__ == '__main__':
    build_mhws_grid('mhws_for_bruno_2019_1458524.h5', 
                    mask_Id=1458524, # 2019
                    vary=False,
                    find=False)