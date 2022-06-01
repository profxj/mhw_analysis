""" Methods for analyzing MHWS """
import sys, os
import numpy as np
import datetime
import pandas
from scipy.io import loadmat

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

def test_matlab():
    #c_file = os.path.join(os.getenv('MHW'), 'db', 'extreme_dy_by_yr_defaults.nc')
    c_file = os.path.join(os.getenv('MHW'), 'db', 'extreme_dy_by_yr_defaults.nc')
    c_ds = xarray.open_dataset(c_file)
    voxels = c_ds.ndays.data[:]
    lat = c_ds.lat.data[:]
    lon = c_ds.lon.data[:]
    lat_grid = np.outer(lat, np.ones(lon.size))
    lon_grid = np.outer(np.ones(lat.size), lon)
    # NWP
    NWP_vox=voxels[361:624, 401:720, :]  # Permuted
    #NWP_vox=voxels[401:720,361:624, :]
    NWP_lon=lon_grid[401:720,361:624]
    NWP_lat=lat_grid[401:720,361:624]

    NWP_mean = np.nanmean(NWP_vox, axis=(0,1))
    embed(header='88 of analy_sys')

def ocean_area_trends(c_file:str, outfile:str):
    # Load up
    c_file = os.path.join(os.getenv('MHW'), 'db', c_file)
    c_ds = xarray.open_dataset(c_file)
    voxels = c_ds.ndays.data[:]
    lat = c_ds.lat.data[:]
    lon = c_ds.lon.data[:]
    lat_grid = np.outer(lat, np.ones(lon.size))
    lon_grid = np.outer(np.ones(lat.size), lon)

    # Table
    df = pandas.DataFrame()

    # NWP
    #NWP_vox=voxels[401:720,361:624, :]
    NWP_vox=voxels[360:624, 400:720, :]  # Permuted
    #NWP_lon=lon_grid[401:720,361:624]
    #NWP_lat=lat_grid[401:720,361:624]
    df['NWP'] = np.nanmean(NWP_vox, axis=(0,1))

    # SP 
    #SP_vox=voxels[721:1172,121:360, :]
    SP_vox=voxels[120:360, 720:1172, :] # Permuted
    df['SP'] = np.nanmean(SP_vox, axis=(0,1))

    # AUS
    #AUS_vox=voxels(:,401:720,121:360);
    AUS_vox=voxels[120:360, 400:720, :] # Permued
    df['AUS'] = np.nanmean(AUS_vox, axis=(0,1))

    # IND
    #IND_vox=voxels(:,81:400,121:480);
    IND_vox=voxels[120:480, 80:400, :] # Permuted
    df['IND'] = np.nanmean(IND_vox, axis=(0,1))

    # SWA
    #SWA_vox=voxels(:,1161:1368,121:360);
    SWA_vox=voxels[120:360, 1160:1368, :]  #  Permuted
    df['SWA'] = np.nanmean(SWA_vox, axis=(0,1))

    # SEA
    #SEA_vox1=voxels(:,1369:1440,121:361);
    SEA_vox1=voxels[120:361, 1368:1440, :] # permuted
    #SEA_vox2=voxels(:,1:80,121:361);
    SEA_vox2=voxels[120:361, 0:80, :] # permuted
    df['SEA'] = (np.nanmean(SEA_vox1, axis=(0,1))  + 
                 np.nanmean(SEA_vox2, axis=(0,1))) / 2.

    # NEA
    #NEA_vox1=voxels(:,1:164,361:632);
    NEA_vox1=voxels[360:632, 0:164, :] # permuted
    #NEA_vox2=voxels(:,1281:1440,361:632);
    NEA_vox2=voxels[360:632, 1280:1440, :] # permuted
    df['NEA'] = (np.nanmean(NEA_vox1, axis=(0,1))  + 
                 np.nanmean(NEA_vox2, axis=(0,1))) / 2.

    # ARC
    #ARC_vox=voxels(:,:,625:720);
    ARC_vox=voxels[624:720, :, :] # Permuted
    df['ARC'] = np.nanmean(ARC_vox, axis=(0,1))

    # ACC
    #ACC_vox=voxels(:,:,1:120);
    ACC_vox=voxels[0:120, :, :] # Permuted
    df['ACC'] = np.nanmean(ACC_vox, axis=(0,1))

    # NEP
    #NEP_vox=voxels(:,721:1128,361:624);
    NEP_vox=voxels[360:624, 720:1128, :] # Permute
    # Mask
    NEP_mask = loadmat('NEP_mask.mat')['NEP_mask']
    NEP_mask = NEP_mask.T
    full_NEP_mask = np.zeros((NEP_mask.shape[0], NEP_mask.shape[1], 
                              NEP_vox.shape[2]))
    for kk in range(NEP_vox.shape[2]):
        full_NEP_mask[:,:,kk] = NEP_mask
    df['NEP'] = np.nanmean(NEP_vox * full_NEP_mask, axis=(0,1))
    
    # NWA
    #NWA_vox=voxels(:,1049:1280,361:625);
    NWA_vox=voxels[360:625, 1048:1280, :] # Permuted
    # Mask
    NWA_mask = loadmat('NWA_mask.mat')['NWA_mask']
    NWA_mask = np.swapaxes(NWA_mask, 0, 1)
    #NWA_mask = np.flip(NWA_mask, 1)
    full_NWA_mask = np.zeros((NWA_mask.shape[0], NWA_mask.shape[1], 
                              NWA_vox.shape[2]))
    for kk in range(NWA_vox.shape[2]):
        full_NWA_mask[:,:,kk] = NWA_mask
    df['NWA'] = np.nanmean(NWA_vox * full_NWA_mask, axis=(0,1))

    # Write
    df.to_csv(outfile)



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

    # Test trend
    if flg_main & (2 ** 3):
        test_matlab()

    # Trend by ocean area
    if flg_main & (2 ** 4):
        ocean_area_trends('extreme_dy_by_yr_defaults.nc',
                          'severe_ocean_areas_orig.csv')


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg_main = 0
        #flg_main += 2 ** 0  # Defaults
        #flg_main += 2 ** 1  # Days by year, 2019 detrend local
        #flg_main += 2 ** 2  # Days by year, 2019 
        #flg_main += 2 ** 3  # Trend
        flg_main += 2 ** 4  # Ocean area analysis
    else:
        flg_main = sys.argv[1]

    main(flg_main)