""" Generate Images for MHW AI analysis"""

import os
import numpy as np
import h5py
import datetime

import xarray

from oceanpy.sst import utils as sst_utils

from mhw_analysis.systems import io as mhwsys_io
from mhw_analysis import utils as mhw_utils

from IPython import embed



def build_intermediate(outfile='MHW_sys_intermediate.npz', xydim=64,
                       mask_start=(1982, 1, 1), full_mask=None,
                       debug=False, Z500_delta_t=5,
                       Z500_xydim=None,
                       subtract_climate=False):
    """
    Build a set of intermeidate (NVox ~ 1000) images for analysis

    And a corresponding null set

    Args:
        outfile:
        xydim (int, optional):
        mask_start:
        Z500_delta_t (int, optional):
            Offset for Z500:  t_Z500 = t - Z500_delta_t
        debug:
        subtract_climate (bool, optional):
            Subtract off the climate?

    Returns:

    """

    # Load systems
    mhw_systems = mhwsys_io.load_systems(vary=True)
    # Intermediate
    sys_1000 = (mhw_systems.max_area > 300) & (mhw_systems.max_area < 5000) & (
            np.abs(mhw_systems.lat) < 65.)
    nint = np.sum(sys_1000)
    int_systems = mhw_systems[sys_1000]

    # SST coords
    lat, lon= sst_utils.noaa_oi_coords()
    dlon, dlat = 0.25, 0.25

    print("We have {} intermediate systems".format(nint))

    # Load Z500
    ncep_path = os.getenv("NCEP_DOE")
    ifile = os.path.join(ncep_path, 'NCEP-DOE_Z500.nc')
    ncep_xr = xarray.load_dataset(ifile)
    Z500_dlon, Z500_dlat = 2.5, 2.5

    # Load climate
    if subtract_climate:
        cfile = os.path.join(ncep_path, 'NCEP-DOE_Z500_climate.nc')
        ncep_climate = xarray.load_dataset(cfile)

    # Times
    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()

    # Final arrays
    times = np.zeros(nint, dtype=int)
    null_times = np.zeros(nint, dtype=int)
    img_arr = np.zeros((xydim, xydim, nint), dtype=int)
    null_arr = np.zeros((xydim, xydim, nint), dtype=int)
    if Z500_xydim is None:
        z500_arr = np.zeros((xydim, xydim, nint), dtype=int)
        z500_null_arr = np.zeros((xydim, xydim, nint), dtype=int)
    else:
        z500_arr = np.zeros((Z500_xydim, Z500_xydim, nint), dtype=int)
        z500_null_arr = np.zeros((Z500_xydim, Z500_xydim, nint), dtype=int)


    # Load mask (50Gb!!)
    if full_mask is None:
        mhw_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf')
        print("Loading the mask: {}".format(mhw_mask_file))
        print("Be patient...")
        full_mask = mhwsys_io.load_full_mask(mhw_mask_file=mhw_mask_file)
        #f = h5py.File(mhw_mask_file, mode='r')
        #full_mask = f['mask'][:,:,:]
        print('Mask loaded')
        #f.close()

    # Loop me
    kk = 0
    for iid, mhw_sys in int_systems.iterrows():
        if debug:
            if iid != 608741:
                continue
        if kk % 100 == 0:
            print('kk={}'.format(kk))
        # Cut on the full system
        #mask = full_mask[:,:,mhw_sys.zboxmin:mhw_sys.zboxmax+1]
        mask = full_mask.isel(time=slice(mhw_sys.zboxmin,mhw_sys.zboxmax))
        # Find max day
        idx = mask.data == mhw_sys.mask_Id
        summ = np.sum(idx, axis=(0,1))
        iarea = np.argmax(summ)
        times[kk] = t0 + mhw_sys.zboxmin+iarea

        # Grab the sub-image
        ii = int(np.round(mhw_sys.xcen))
        jj = int(np.round(mhw_sys.ycen))
        lat_ii = lat[ii]
        lon_jj = lon[jj]
        # Deal with wrap around
        lons = [lon_jj-dlon*xydim//2, lon_jj+dlon*(xydim//2-1)]
        lats = [lat_ii-dlat*xydim//2, lat_ii+dlat*(xydim//2-1)]
        if (lats[0] < -90.) or (lats[1] > 90.):
            import pdb; pdb.set_trace()
        sub_img = mhw_utils.grab_geo_subimg(mask, lats, lons)
        #i0 = ii - xydim // 2
        #i1 = ii + xydim // 2
        #j0 = jj - xydim // 2
        #j1 = jj + xydim // 2
        ## ii
        #if i0 < 0:
        #    # Better not get here
        #    import pdb;  pdb.set_trace()
        #    rolli = xydim
        #    i0 += xydim
        #    i1 += xydim
        #elif i1 > mask.shape[0]-1:
        #    # Better not get here
        #    import pdb;  pdb.set_trace()
        #    rolli = -1*xydim
        #    i0 -= xydim
        #    i1 -= xydim
        #else:
        #    rolli = 0
        ## jj
        #if j0 < 0:
        #    rollj = xydim
        #    j0 += xydim
        #    j1 += xydim
        #    jrolling=True
        #elif j1 > mask.shape[1]-1:
        #    rollj = -1*xydim
        #    j0 -= xydim
        #    j1 -= xydim
        #    jrolling=True
        #else:
        #    rollj = 0
        #    jrolling=False
        # Setup
        #mask = np.roll(mask, (rolli, rollj), axis=(0,1))
        #img_arr[:,:,kk] = mask[i0:i1, j0:j1, iarea]

        # Fill
        img_arr[:,:,kk] = sub_img.isel(time=iarea).data

        # debug
        jrolling = False
        if jrolling and kk==47 and debug:# or kk==47:
            from matplotlib import  pyplot as plt
            plt.clf()
            plt.imshow(img_arr[:,:,kk])
            plt.show()
            import pdb; pdb.set_trace()

        # Null -- Scan +/- 5 years
        n_null = 999999
        for year_off in range(-5, 6):
            if year_off == 0:
                continue
            # Grab the new date
            tdate = datetime.date.fromordinal(times[kk])
            # Deal with leap day
            if tdate.month == 2 and tdate.day == 29:
                tdate = datetime.date.fromordinal(times[kk]-1)
            tnull = datetime.date(tdate.year+year_off,
                                  tdate.month, tdate.day).toordinal()
            inull = tnull - t0
            # Is it in the analysis window?
            if inull < 365 or inull >= full_mask.shape[2]:  # Skip 1982
                continue
            # Slurp and shuffle
            #null_mask = full_mask[:,:,inull]
            #null_mask = np.roll(null_mask, (rolli, rollj), axis=(0,1))
            #tmp = null_mask[i0:i1, j0:j1]
            null_sub = full_mask.isel(time=inull)
            null_sub = mhw_utils.grab_geo_subimg(null_sub, lats, lons)
            tmp = null_sub.data
            # Is the image more blank?
            if np.sum(tmp > 0) < n_null:
                n_null = np.sum(tmp > 0)
                null_arr[:, :, kk] = tmp
                null_times[kk] = tnull
        # Warning if not very blank
        if n_null > 300:
            print("kk=={}: Null image has {}.  Be warned!".format(kk, n_null))

        # Z500 -- Good and null
        for out_arr, mhw_time in zip([z500_arr, z500_null_arr], [times[kk], null_times[kk]]):
            # Good first
            time = datetime.datetime.fromordinal(mhw_time-Z500_delta_t)
            ds = ncep_xr.sel(time=time)
            # Subtract climate?
            if subtract_climate:
                day_of_year = mhw_utils.doy(time)
                climate = ncep_climate.sel(doy=day_of_year)
                ds = (ds.Z500 - climate.seasonalZ500).to_dataset(name='Z500')

            if Z500_xydim is None:
                # Expand
                ds_out = ds.interp(lat=lat, lon=lon)
                # Fill Nans with average between edges
                bad_lon = np.where(lon > max(ds.lon.values))[0]
                fill_val = (ds_out.Z500.data[:,min(bad_lon)-1] + ds_out.Z500.data[:,0])/2
                for yy in bad_lon:
                    ds_out.Z500.data[:,yy] = fill_val
                # Sub image
                #out_arr[:, :, kk] = np.roll(ds_out.Z500.data, (rolli, rollj), axis=(0, 1))[i0:i1, j0:j1]
                out_arr[:, :, kk] = mhw_utils.grab_geo_subimg(ds_out.Z500, lats, lons).data
            else:
                Z500_lon = ds.lon.values[np.argmin(np.abs(lon_jj-ds.lon.values))]
                Z500_lat = ds.lat.values[np.argmin(np.abs(lat_ii-ds.lat.values))]
                Z500_lons = [Z500_lon - Z500_dlon * Z500_xydim // 2,
                             Z500_lon + Z500_dlon * (Z500_xydim // 2-1)]
                Z500_lats = [Z500_lat + Z500_dlat * Z500_xydim // 2,
                             Z500_lat - Z500_dlat * (Z500_xydim // 2-1)]
                # Warning, this image is flipped in latitude from SST!
                out_arr[:, :, kk] = mhw_utils.grab_geo_subimg(ds.Z500, Z500_lats, Z500_lons).data
            if jrolling and debug:
                from matplotlib import  pyplot as plt
                plt.clf()
                plt.imshow(out_arr[:,:,kk])
                plt.show()
                import pdb; pdb.set_trace()

        '''
        # Null
        time=datetime.datetime.fromordinal(null_times[kk]-Z500_delta_t)
        day_of_year = mhw_utils.doy(time)
        ds = ncep_xr.sel(time=time)
        if subtract_climate:
            climate = ncep_climate.sel(day=day_of_year)
            ds = (ds.Z500 - climate.seasonalZ500).to_dataset(name='Z500')

        ds_out = ds.interp(lat=lat, lon=lon)
        fill_val = (ds_out.Z500.data[:,min(bad_lon)-1] + ds_out.Z500.data[:,0])/2
        for yy in bad_lon:
            ds_out.Z500.data[:,yy] = fill_val
        z500_null_arr[:,:,kk] = np.roll(ds_out.Z500.data, (rolli, rollj), axis=(0, 1))[i0:i1, j0:j1]
        '''

        # Increment
        kk += 1
        embed(header='234 of images')

    # Add times
    int_systems['max_time'] = times
    int_systems['null_time'] = null_times

    # Save
    if debug:
        import pdb; pdb.set_trace()
    np.savez_compressed(outfile, images=img_arr, null=null_arr, z500=z500_arr, z500_null=z500_null_arr)
    int_systems.to_hdf(outfile.replace('npz', 'hdf'), 'mhw_sys', mode='w')


# Testing
if __name__ == '__main__':
    # Debug
    if True:
        print("Loading debug full_mask")
        mask = mhwsys_io.load_mask_from_dates(
            (1982, 1, 1), (1985, 1, 1),
            mhw_mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf'))
        #full_mask = mask.data
        build_intermediate(full_mask=mask, debug=True, Z500_xydim=16) # subtract_climate=True)

    # Real deal
    if False:
        if False:
            import os, h5py
            import numpy as np
            mhw_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf')
            f = h5py.File(mhw_mask_file, mode='r')
            print("Loading the mask: {}".format(mhw_mask_file))
            full_mask = f['mask'][:,:,:]
            f.close()
            print('Mask loaded')
        build_intermediate(full_mask=full_mask, debug=False)

    # Subtract climate
    if False:
        build_intermediate(outfile='MHW_sys_intermediate_climate.npz',
                       full_mask=full_mask, debug=False, subtract_climate=True)

    # Subtract climate + larger
    if False:
        build_intermediate(outfile='MHW_sys_intermediate_climate.npz',
                           full_mask=full_mask, debug=False, subtract_climate=True)
