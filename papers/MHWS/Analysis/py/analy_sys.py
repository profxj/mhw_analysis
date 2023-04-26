""" Methods for analyzing MHWS """
import sys, os
import numpy as np
import datetime
import pandas
from scipy.io import loadmat

import xarray

from mhw_analysis.systems import io as mhw_sys_io
from mhw_analysis.systems import analysisc
from mhw_analysis.systems import defs

from IPython import embed 

# Local
sys.path.append(os.path.abspath("../Analysis/py"))

# Regions
regions = {}
regions['NWP'] = dict(lat=(360,624), lon=(400,720))
regions['AUS'] = dict(lat=(120,360), lon=(400,720))
regions['IND'] = dict(lat=(120,480), lon=(80,400))
regions['ARC'] = dict(lat=(624,720), lon=(0,1440))
regions['NEA'] = dict(lat=(360,632), lon=(1280,164)) # Wrapping
regions['NEP'] = dict(lat=(360,624), lon=(720,1128)) 
regions['NWA'] = dict(lat=(360,625), lon=(1048,1280)) 
regions['SEA'] = dict(lat=(120,361), lon=(1368,80)) # Wrapping
regions['SWA'] = dict(lat=(120,360), lon=(1160,1368)) 
regions['SP'] = dict(lat=(120,360), lon=(720,1172)) 
regions['ACC'] = dict(lat=(0,120), lon=(0,1440)) 

def vox_region(region, voxels):
    if regions[region]['lon'][0] < regions[region]['lon'][1]:
        return [voxels[regions[region]['lat'][0]:regions[region]['lat'][1],
            regions[region]['lon'][0]:regions[region]['lon'][1],:]]
    else:
        vox1 =  voxels[regions[region]['lat'][0]:regions[region]['lat'][1],
            regions[region]['lon'][0]:]
        vox2 =  voxels[regions[region]['lat'][0]:regions[region]['lat'][1],
            0:regions[region]['lon'][1]]
        return [vox1, vox2]

def process_vox(lat_grid, voxels, region, df, 
                nvox_dict, lat_dict, masks_dict, 
                debug=False):
    vox_list = vox_region(region, voxels)
    # Count em
    nvoxs, means, mean_lats = [], [], []
    for vox in vox_list:
        nvox = np.sum(np.isfinite(vox))
        if region in masks_dict.keys():
            if debug:
                embed(header='63 of analy')
            # Transpose
            mask = masks_dict[region].T
            full_mask = np.zeros((mask.shape[0], mask.shape[1], vox.shape[2]))
            for kk in range(vox.shape[2]):
                full_mask[:,:,kk] = mask
                mean_days = np.nanmean(vox * full_mask, axis=(0,1))
        else:
            mean_days = np.nanmean(vox, axis=(0,1))
        # Lats
        lats = lat_grid[regions[region]['lat'][0]:regions[region]['lat'][1],...]
        mean_lat = np.nanmean(lats)
        #
        nvoxs.append(nvox)
        means.append(mean_days)
        mean_lats.append(mean_lat)
    # Sum em
    nvox_dict[region] = np.sum(nvoxs)
    # Weighted mean for days
    warrays = [item*nvox for item,nvox in zip(means, nvoxs)]
    df[region] = np.sum(np.array(warrays), axis=0) / np.sum(nvoxs)
    # Latitudes
    warrays = [item*nvox for item,nvox in zip(mean_lats, nvoxs)]
    lat_dict[region] = np.sum(np.array(warrays), axis=0) / np.sum(nvoxs)
    return
    


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
    nvox_dict = {}
    lat_dict = {}

    # NWP
    #NWP_vox=voxels[401:720,361:624, :]
    #NWP_vox=voxels[360:624, 400:720, :]  # Permuted

    #NWP_vox= vox_region('NWP', voxels)[0]
    #df['NWP'] = np.nanmean(NWP_vox, axis=(0,1))

    #process_vox(lat_grid, voxels, 'NEA', df, nvox_dict, lat_dict)

    # Masks
    masks_dict = {}
    masks_dict['NEP'] = loadmat('NEP_mask.mat')['NEP_mask']
    masks_dict['NWA'] = loadmat('NWA_mask.mat')['NWA_mask']

    #process_vox(lat_grid, voxels, 'NEP', df, nvox_dict, lat_dict, masks_dict, debug=True)

    # Do em all!
    for region in regions.keys():
        process_vox(lat_grid, voxels, region, df, nvox_dict, lat_dict, masks_dict)
        
    # Add all
    warrays, weights = [], []
    for region in regions.keys():
        weight = nvox_dict[region] * np.cos(lat_dict[region]*np.pi/180.)
        warray = df[region].values * weight
        weights.append(weight)
        warrays.append(warray)

    df['ALL'] = np.sum(np.array(warrays), axis=0) / np.sum(weights)

    '''
    # SP 
    #SP_vox=voxels[721:1172,121:360, :]
    #SP_vox=voxels[120:360, 720:1172, :] # Permuted
    SP_vox= vox_region('SP', voxels)
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
    embed(header='171 of analy_sys')
    
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
    '''

    # Write
    df.to_csv(outfile)
    print(f'Wrote: {outfile}')

def calc_misc_year_metrics():
    # Do the stats
    years = 1983 + np.arange(37)

    # Load MHWE
    MHW_path = os.getenv('MHW')
    MHWE_file = 'mhw_events_allsky_2019.parquet'
    mhwe_file = os.path.join(MHW_path, 'db', MHWE_file)
    print('Loading MHWE')
    mhwe = pandas.read_parquet(mhwe_file)
    print(f"There are {len(mhwe)} MHWEs")
    print('Done')
    # Need to build the end date
    #  data is the same time_start
    #  Just add days

    # #############
    # MHWS 
    mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019.csv')
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=False)
    # Convert to days
    tdur_days = mhw_sys.duration.values / np.timedelta64(1, 'D')
    mhw_sys['duration'] = tdur_days
    mhw_sys['avg_area'] = mhw_sys.NVox_km.values / tdur_days

    # Stats
    mean_avg_area = []
    median_avg_area = []
    mean_wgt_avg_area = []
    mean_t_duration = []
    mean_wgt_t_duration = []
    median_t_duration = []
    MHWS_nstart = []
    MHWE_nstart = []
    for jj, year in enumerate(years):

        # Identify those in the year
        day1 = np.datetime64(datetime.datetime(year,1,1))
        day1n = np.datetime64(datetime.datetime(year+1,1,1))
        day_beg = np.maximum(mhw_sys.startdate, day1)
        day_end = np.minimum(mhw_sys.enddate,
                             np.datetime64(datetime.datetime(year+1,1,1)))  # Should subtract a day
        ndays = day_end-day_beg
        in_year = ndays > datetime.timedelta(days=0)
        days = ndays / np.timedelta64(1, 'D')

        # Average areas
        mean_avg_area.append(np.mean(mhw_sys.avg_area[in_year]))
        median_avg_area.append(np.median(mhw_sys.avg_area[in_year]))
        # Weighted average areas
        mean_wgt_avg_area.append(np.sum(mhw_sys.avg_area[in_year]*days[in_year]) /
                                 np.sum(days[in_year]))

        # Durations
        mean_t_duration.append(np.mean(mhw_sys.duration[in_year]))
        median_t_duration.append(np.median(mhw_sys.duration[in_year]))
        # Weighted average duration
        mean_wgt_t_duration.append(np.sum(mhw_sys.duration[in_year]*days[in_year]) /
                                 np.sum(days[in_year]))

        # Starts
        start = (mhw_sys.startdate >= day1) & in_year
        MHWS_nstart.append(np.sum(start))

        # MHWE
        mhwe_start = (mhwe.date >= day1) & (mhwe.date < day1n)
        MHWE_nstart.append(np.sum(mhwe_start))


    # Write
    df = pandas.DataFrame()
    df['year'] = years
    df['mean_avg_area'] = mean_avg_area
    df['mean_wgt_avg_area'] = mean_wgt_avg_area
    df['median_avg_area'] = median_avg_area
    df['mean_t_duration'] = mean_t_duration
    df['mean_wgt_t_duration'] = mean_wgt_t_duration
    df['median_t_duration'] = median_t_duration
    df['MHWS_nstart'] = MHWS_nstart
    df['MHWE_nstart'] = MHWE_nstart

    outfile = 'mhw_stats_by_year_2019.csv'
    df.to_csv(outfile)
    print(f'Wrote: {outfile}')

def calc_area(lat):
    R_Earth = 6371.0
    cell_km = 0.25 * 2 * np.pi * R_Earth / 360.

    # 
    cell_lat = cell_km * cell_km * np.cos(np.pi * lat / 180.);

    return cell_lat

def calc_prediction():
    """
    """
    # #############
    # MHWS 
    mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019.csv')
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=False)

    # Mask
    day_earlier = (2019,5,1)
    day_predict = (2019,6,1)
    dt_earlier = np.datetime64(datetime.datetime(
        day_earlier[0], day_earlier[1], day_earlier[2]))
    dt_predict = np.datetime64(datetime.datetime(
        day_predict[0], day_predict[1], day_predict[2]))

    mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_mask.nc')
    mask_da = mhw_sys_io.load_mask_from_dates(day_earlier, day_predict, 
                                           mhw_mask_file=mask_file) 

    mask_predict = mask_da.sel(time=dt_predict).data[:]
    active = np.where(mask_predict > 0)
    a_lat = mask_da.lat[active[0]]

    cell_area = calc_area(a_lat.data)
    area_predict = np.sum(cell_area)
    print(f"Predicted area: {area_predict} km^2")

    # Now query MHWS
    mask_earlier = mask_da.sel(time=dt_earlier).data[:]
    mask_good_earlier = np.zeros_like(mask_earlier)
    mask_modsev_earlier = np.zeros_like(mask_earlier)
    mask_sev_earlier = np.zeros_like(mask_earlier)
    for iMHWS in np.unique(mask_predict)[1:]:  # Skip the first one which is 0
        in_earlier = np.where(mask_earlier == iMHWS)
        mask_good_earlier[in_earlier] = iMHWS
        # Check severity
        idx = np.where(mhw_sys.mask_Id == iMHWS)[0][0]
        if mhw_sys.iloc[idx].NVox_km > defs.type_dict_km[defs.classb][0]:
            mask_modsev_earlier[in_earlier] = iMHWS
        if mhw_sys.iloc[idx].NVox_km > defs.type_dict_km[defs.classc][0]:
            mask_sev_earlier[in_earlier] = iMHWS


    # % 1 month earlier
    active_earlier = np.where(mask_good_earlier > 0)
    a_lat = mask_da.lat[active_earlier[0]]
    cell_area = calc_area(a_lat.data)
    area_earlier = np.sum(cell_area)

    # % 1 month earlier and moderate/severe
    active_earlier = np.where(mask_modsev_earlier > 0)
    a_lat = mask_da.lat[active_earlier[0]]
    cell_area = calc_area(a_lat.data)
    area_modsev_earlier = np.sum(cell_area)

    # % 1 month earlier and severe
    active_earlier = np.where(mask_sev_earlier > 0)
    a_lat = mask_da.lat[active_earlier[0]]
    cell_area = calc_area(a_lat.data)
    area_sev_earlier = np.sum(cell_area)


    print(f"Of the MHWS that are active on {day_predict}, {area_earlier/area_predict} were active on {day_earlier}")
    print(f"And {area_modsev_earlier/area_earlier} were moderate or severe")
    print(f"And {area_sev_earlier/area_earlier} were severe")

    embed(header='377 of anly sys')

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
        # Original
        #ocean_area_trends('extreme_dy_by_yr_defaults.nc',
        #                  'severe_ocean_areas_orig.csv')

        # Severe
        ocean_area_trends('severe_km_dy_by_yr_2019.nc',
                          'ChangePoint/severe_ocean_areas_2019.csv')
        # Moderate
        #ocean_area_trends('moderate_km_dy_by_yr_2019.nc',
        #                  'moderate_ocean_areas_2019.csv')
        # Minor 
        #ocean_area_trends('minor_km_dy_by_yr_2019.nc',
        #                  'minor_ocean_areas_2019.csv')

    # Misc year metrics
    if flg_main & (2 ** 5):
        calc_misc_year_metrics()

    # Prediction
    if flg_main & (2 ** 6):
        calc_prediction()


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg_main = 0
        #flg_main += 2 ** 0  # Defaults
        #flg_main += 2 ** 1  # Days by year, 2019 detrend local
        #flg_main += 2 ** 2  # Days by year, 2019 
        #flg_main += 2 ** 3  # Trend
        #flg_main += 2 ** 4  # Ocean area analysis
        #flg_main += 2 ** 5  # Misc year metrics
        flg_main += 2 ** 6  # MHWS prediction
    else:
        flg_main = sys.argv[1]

    main(flg_main)