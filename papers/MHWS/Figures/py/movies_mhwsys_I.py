""" Figures for the first paper on MHW Systems"""
import os, sys
import numpy as np
from pkg_resources import resource_filename

# Parallel processing

from datetime import date

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.animation as animation

mpl.rcParams['font.family'] = 'splttixgeneral'

import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import healpy as hp

import pandas

#import iris
#import iris.quickplot as qplt


from oceanpy.sst import io as sst_io

from mhw_analysis.systems import io as mhw_sys_io

from IPython import embed

mhw_path = '/home/xavier/Projects/Oceanography/MHW/db'
noaa_path = os.getenv('NOAA_OI')

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import defs, analy_utils, fitting

cm_heat = plt.get_cmap('YlOrRd')

def drawmap(ax, data):
    data.plot.pcolormesh(add_colorbar=False, cmap=cm_heat,
        vmin=0., vmax=3., transform=ccrs.PlateCarree())
    #ax.set_title(data.title)
    ax.coastlines()
    #ax.gridlines()
    
def myanimate(i, ax, mask_da, isys, lat_mnx, lon_mnx, dt):
    if (i%10) == 0:
        print("Image: {}".format(i))
    ax.clear()
    # Load up
    itime = isys.startdate+pandas.Timedelta('{}days'.format(i*dt))
    mask_cut = mask_da.sel(
            time=slice(itime, itime+pandas.Timedelta('1days')),
            lat=slice(lat_mnx[0], lat_mnx[1]),
            lon=slice(lon_mnx[0], lon_mnx[1]))
    mask = mask_cut.data[:,:,0]
    # In mask
    in_mask = mask == isys.mask_Id
    # Load SST
    sst = sst_io.load_noaa(
        (itime.day,itime.month,itime.year), 
        subtract_seasonal=True,
        climate_file=os.path.join(os.getenv('NOAA_OI'), 
        'NOAA_OI_climate_1983-2019.nc'))

    sst = sst.sel(lat=slice(lat_mnx[0], lat_mnx[1]), 
                lon=slice(lon_mnx[0], lon_mnx[1]))
    sst.data[~in_mask] = np.nan
    #
    new_image = drawmap(ax, sst)


def extreme_movie(sys_file, mask_file, ordinal, outfile, 
                          debug=False, cold=False, dt=5,
                          lon_mnx=(0., 360.), lat_mnx=(-90.,90.)):
    """
    Snapshots of an extreme MHWS
    

    Args:
        sys_file:
        mask_file:
        outroot:
        next (int):
            Number of examples in decreasing order of NVox
        debug:

    Returns:

    """
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(
        mhw_sys_file=os.path.join(mhw_path, sys_file))

    # Grab the extreme systems
    isrt = np.argsort(mhw_sys.NVox.values)

    # Grab it
    isys = mhw_sys.iloc[isrt[-1-ordinal]]
    
    # Frames
    tot_days = (isys.enddate -isys.startdate).days
    nframes = tot_days // dt
    print("Movie will contain nframes={}".format(nframes))

    # Grab the mask
    mask_da = mhw_sys_io.load_mask_from_system(
        isys, mhw_mask_file=os.path.join(mhw_path, mask_file))

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Grace data', artist='CEED workshop',
                    comment='Movie for sequence of images')
    writer = FFMpegWriter(fps=2, metadata=metadata)

    # Create the figure

    fig = plt.figure(figsize=[20,10])  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=isys.lon))
        
    ani = animation.FuncAnimation(fig, myanimate, 
                                  frames=np.arange(nframes), 
                                  fargs=(ax, mask_da, isys, lat_mnx, lon_mnx,
                                         dt), 
                                  interval=100)
    ani.save(outfile, writer=writer)
    print("Wrote: {}".format(outfile))



#### ########################## #########################
def main(flg_movie):
    if flg_movie == 'all':
        flg_movie = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_movie = int(flg_movie)

    if flg_movie & (2 ** 0):
        extreme_movie('MHW_systems_vary.csv', 'MHW_mask_vary.nc', 
            0, 'fig_extreme_vary_ex0.mp4', lon_mnx=(120., 300.),
            lat_mnx=(-70., 70.))

    if flg_movie & (2 ** 1):
        extreme_movie('MHWS_2019.csv', 'MHWS_2019_mask.nc',
            0, 'fig_extreme_ex0.mp4', lon_mnx=(120., 300.),
            lat_mnx=(-70., 70.))


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_movie = 0
        #flg_movie += 2 ** 0  # MHW Events (spatial)
        flg_movie += 2 ** 1  # MHWS movie; 2019
    else:
        flg_movie = sys.argv[1]

    main(flg_movie)
