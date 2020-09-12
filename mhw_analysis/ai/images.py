""" Generate Images for MHW AI analysis"""

import os
import numpy as np
import h5py
import datetime

#import iris
import xarray
import xesmf

from mhw_analysis.systems import io as mhwsys_io

from IPython import embed


def grab_z500(cube, lon, lat, width):
    """
    Grab Z500 data around the input lon, lat
    with the input width

    Parameters
    ----------
    cube : iris.Cube
    lon : float (deg)
    lat : float (deg)
    width : float (deg)

    Returns
    -------
    numpy.ndarray

    """

    # Generate a constraint
    #  WARNING -- This may not work across the 0-360deg boundary
    constraint = iris.Constraint(latitude=lambda cell: (lat-width/2) <= cell <= (lat+width/2),
                                 longitude=lambda cell: (lon - width/2) <= cell <= (
                                         lon + width/2))
    # Apply
    sub_cube = cube.extract(constraint)

    # Return the data
    return sub_cube.data[:]


def build_intermediate(outfile='MHW_sys_intermediate.npz', xydim=64,
                       mask_start=(1982, 1, 1), full_mask=None,
                       debug=False, Z500_delta_t=5):
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
    lat = -89.875 + np.arange(720) * 0.25
    lon = 0.125 + np.arange(1440) * 0.25

    print("We have {} intermediate systems".format(nint))

    # Load Z500
    ncep_path = os.getenv("NCEP_DOE")
    ifile = os.path.join(ncep_path, 'NCEP-DOE_Z500.nc')
    # Load cube
    ncep_xr = xarray.load_dataset(ifile)

    # Times
    t0 = datetime.date(mask_start[0], mask_start[1], mask_start[2]).toordinal()

    # Final arrays
    times = np.zeros(nint, dtype=int)
    null_times = np.zeros(nint, dtype=int)
    img_arr = np.zeros((xydim, xydim, nint), dtype=int)
    null_arr = np.zeros((xydim, xydim, nint), dtype=int)
    z500_arr = np.zeros((xydim, xydim, nint), dtype=int)
    z500_null_arr = np.zeros((xydim, xydim, nint), dtype=int)

    # Load mask (50Gb!!)
    if full_mask is None:
        mhw_mask_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf')
        f = h5py.File(mhw_mask_file, mode='r')
        print("Loading the mask: {}".format(mhw_mask_file))
        full_mask = np.array(f['mask'][:,:,:])
        print('Mask loaded')
        f.close()

    # Loop me
    kk = 0
    for iid, mhw_sys in int_systems.iterrows():
        if debug:
            if iid != 608741:
                continue
        if kk % 100 == 0:
            print('kk={}'.format(kk))
        # Cut on the full system
        mask = full_mask[:,:,mhw_sys.zboxmin:mhw_sys.zboxmax+1]
        # Find max day
        idx = mask == mhw_sys.mask_Id
        summ = np.sum(idx, axis=(0,1))
        iarea = np.argmax(summ)
        times[kk] = t0 + mhw_sys.zboxmin+iarea
        # Grab the sub-image
        ii = int(np.round(mhw_sys.xcen))
        jj = int(np.round(mhw_sys.ycen))
        # Deal with wrap around
        i0 = ii - xydim // 2
        i1 = ii + xydim // 2
        j0 = jj - xydim // 2
        j1 = jj + xydim // 2
        # ii
        if i0 < 0:
            rolli = xydim
            i0 += xydim
            i1 += xydim
        elif i1 > mask.shape[0]-1:
            rolli = -1*xydim
            i0 -= xydim
            i1 -= xydim
        else:
            rolli = 0
        # jj
        if j0 < 0:
            rollj = xydim
            j0 += xydim
            j1 += xydim
        elif j1 > mask.shape[1]-1:
            rollj = -1*xydim
            j0 -= xydim
            j1 -= xydim
        else:
            rollj = 0
        # Setup
        mask = np.roll(mask, (rolli, rollj), axis=(0,1))
        img_arr[:,:,kk] = mask[i0:i1, j0:j1, iarea]
        if debug:# or kk==47:
            from matplotlib import  pyplot as plt
            plt.clf()
            plt.imshow(img_arr[:,:,kk])
            plt.show()
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
            null_mask = full_mask[:,:,inull]
            null_mask = np.roll(null_mask, (rolli, rollj), axis=(0,1))
            tmp = null_mask[i0:i1, j0:j1]
            # Is the image more blank?
            if np.sum(tmp > 0) < n_null:
                n_null = np.sum(tmp > 0)
                null_arr[:, :, kk] = tmp
                null_times[kk] = tnull
        # Warning if not very blank
        if n_null > 300:
            print("kk=={}: Null image has {}.  Be warned!".format(kk, n_null))

        # Z500
        # Good first
        ds = ncep_xr.sel(time=datetime.datetime.fromordinal(times[kk]-Z500_delta_t))
        ds_out = xarray.Dataset({'lat': (['lat'], lat[ii - xydim // 2:ii + xydim // 2]),
                             'lon': (['lon'], lon[jj - xydim // 2:jj + xydim // 2]),
                             }
                            )
        regridder = xesmf.Regridder(ds, ds_out, 'bilinear')
        regridder.clean_weight_file()
        dr_out = regridder(ds['Z500'])
        z500_arr[:,:,kk] = dr_out.data

        # Null
        ds = ncep_xr.sel(time=datetime.datetime.fromordinal(null_times[kk]-Z500_delta_t))
        dr_out = regridder(ds['Z500'])
        z500_null_arr[:,:,kk] = dr_out.data
        embed(header='198 of images')

        # Increment
        kk += 1

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
    if False:
        print("Loading debug full_mask")
        mask = mhwsys_io.load_mask_from_dates(
            (1982, 1, 1), (1985, 1, 1),
            mhw_mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHW_mask_vary.hdf'))
        full_mask = mask.data
        build_intermediate(full_mask=full_mask, debug=True)

    # Real deal
    if True:
        build_intermediate(debug=False)
