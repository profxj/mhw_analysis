""" Script to show an MHW System"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Show an MHW System')
    parser.add_argument("dataset", type=str, help="MHW System set:  orig, vary")
    parser.add_argument("plot_type", type=str, help="Plot type:  first_day")
    parser.add_argument("-m", "--maskid", type=int, help="Mask Id")

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
    import warnings
    import datetime
    from matplotlib import pyplot as plt

    import xarray

    from mhw_analysis.systems import io as mhw_sys_io

    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    vary = False
    if pargs.dataset == 'orig':
        vary=False
    elif pargs.dataset == 'vary':
        vary=True
    else:
        raise IOError("Bad flavor!")

    # Load the systems
    mhw_systems = mhw_sys_io.load_systems(vary=vary)

    # Grab the system
    if pargs.maskid is not None:
        idx = np.where(mhw_systems.mask_Id == pargs.maskid)[0][0]
        mhw_system = mhw_systems.iloc[idx]
    else:
        raise IOError("Must use --maskid for now")

    # Date
    start_date = datetime.date.fromordinal(datetime.date(1982,1,1).toordinal() + mhw_system.zboxmin)
    print("Start date: {}".format(start_date))

    # Grab the mask (this can get big!)
    mask_cube = mhw_sys_io.load_mask_from_system(mhw_system, vary=vary)

    # Plot
    if pargs.plot_type == 'first_day':
        sys_idx = mask_cube.data == mhw_system.mask_Id
        mask_cube.data[np.logical_not(sys_idx)] = 0
        mask_cube.data[sys_idx] = 1
        # Date
        for kk in range(mask_cube.data.shape[2]):
            mask_cube.data[:, :, kk] *= kk + 1
        #
        mask_cube.data[mask_cube.data == 0] = 9999999
        tstart = np.min(mask_cube.data, axis=2).astype(float)
        tstart[tstart == 9999999] = np.nan

        # DataArray for plotting
        embed(header='62 of show')
        tstart_da = xarray.DataArray(tstart, coords=[mask_cube.lat, mask_cube.lon])
        
        # Plot me
        # http://xarray.pydata.org/en/stable/examples/visualization_gallery.html
        fig = plt.figure(figsize=(10, 6))
        plt.clf()

        proj = ccrs.PlateCarree(central_longitude=-180.0)
        #ax = plt.gca(projection=proj)

        # Pacific events
        # Draw the contour with 25 levels.
        cm = plt.get_cmap('rainbow')
        p = tstart_da.plot(cmap=cm, transform=ccrs.PlateCarree(),
                       subplot_kws={'projection': proj},
                           cbar_kwargs={'label': 't_start (Days since)'})
        ax = p.axes

        # Add coastlines to the map created by contourf.
        ax.coastlines()

        # Gridlines
        # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
        gl = ax.gridlines(crs=proj, #ccrs.PlateCarree(),
                          linewidth=2, color='black', alpha=0.5,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right = False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        # gl.xlocator = mticker.FixedLocator([-180., -170., -160, -150., -140, -120, -60, -20.])
        # gl.ylocator = mticker.FixedLocator([30., 40., 50., 60.])


        plt.show()




