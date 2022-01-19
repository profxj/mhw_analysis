""" Figures for the first paper on MHW Systems"""
import os, sys
import numpy as np
from pkg_resources import resource_filename

from datetime import date

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, use
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import matplotlib.animation as animation

mpl.rcParams['font.family'] = 'splttixgeneral'

import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

import healpy as hp

import pandas
import datetime

#import iris
#import iris.quickplot as qplt

import xarray

from oceanpy.sst import io as sst_io
from oceanpy.utils import catalog

from mhw_analysis.systems import io as mhw_sys_io
from mhw_analysis.systems import utils as mhw_sys_utils
from mhw_analysis.systems import analysisc as mhw_analysisc

#from ulmo import io as ulmo_io # for s3

from IPython import embed

mhw_path = '/home/xavier/Projects/Oceanography/MHW/db'
noaa_path = os.getenv('NOAA_OI')

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import defs, analy_utils, fitting


def fig_mhw_events(outfile, mhw_events=None, events=None, duration=None,
                   events_file=None, show_pacific=False):
    """
    Spatial distribution of MHW events

    Args:
        outfile:
        mhw_events:
        events (xarray.DataArray):
        cube:

    Returns:

    """

    if events is None:
        if mhw_events is None:
            embed(header='58 get this right')
            '''
            mhw_file = '/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db'
            engine = sqlalchemy.create_engine('sqlite:///' + mhw_file)
            # Load
            mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                           columns=['date', 'lon', 'lat', 'duration',
                                                    'ievent', 'time_start', 'index', 'category'])
            mhw_events.set_index('date')
            '''

        # Load NOAA for coords
        sst_1982 = sst_io.load_noaa((1,1,1983))

        # Events
        lat_coord = sst_1982.coord('latitude')
        lon_coord = sst_1982.coord('longitude')

        # Cut on duration?
        if duration is not None:
            sub_events = mhw_events[mhw_events.duration > duration]
        else:
            sub_events = mhw_events

        embed(header='81 of figs_mhwsys')

        i_idx = catalog.match_ids(sub_events['lat'], lat_coord.points, require_in_match=True)
        j_idx = catalog.match_ids(sub_events['lon'], lon_coord.points, require_in_match=True)

        n_events = np.zeros((lat_coord.shape[0], lon_coord.shape[0]))
        for ii, jj in zip(i_idx, j_idx):
            n_events[ii, jj] += 1

        # Cube it
        cube = iris.cube.Cube(n_events, var_name='N_events',
                              dim_coords_and_dims=[(lat_coord, 0),
                                                   (lon_coord, 1)])
        # Save?
        # REFACTOR FOR XARRAY
        if cube_file is not None:
            iris.save(cube, cube_file, zlib=True)

    # Pacific?
    if show_pacific:
        # REFACTOR FOR XARRAY
        import pdb; pdb.set_trace()
        #latlon_constraint = iris.Constraint(
        #    latitude=lambda cell: 0. <= cell < 50.,
        #    longitude=lambda cell: 190. < cell < 250.)
        #cube_slice = cube.extract(latlon_constraint)
    else:
        event_slice = events

    fig = plt.figure(figsize=(7, 5))
    plt.clf()

    proj = ccrs.PlateCarree(central_longitude=-180.0)
    ax = plt.gca(projection=proj)

    # Pacific events
    # Draw the contour with 25 levels.
    cm = plt.get_cmap('YlOrRd')
    p = event_slice.plot(cmap=cm, transform=ccrs.PlateCarree(),
                       subplot_kws={'projection': proj},
                       cbar_kwargs={'label': 'Number of MHW Events with t>1 month',
                                    'fraction': 0.020, 'pad': 0.04})
    ax = p.axes
    #cplt = iris.plot.contourf(cube_slice, 10, cmap=cm)  # , vmin=0, vmax=20)#, 5)
    #cb = plt.colorbar(cplt, fraction=0.020, pad=0.04)
    #cb.set_label('Average Annual Number of MHW Events')
    #cb.set_label('Number of MHW Events with t>1 month')

    # Gridlines
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5,
                      linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
    gl.xlocator = mticker.FixedLocator([-240., -180., -120, -60, 0, 60, 120.])
    #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])

    # Add coastlines to the map created by contourf.
    ax.coastlines()

    # Turn off Title
    plt.title('')

    # Layout and save
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_mhw_events_time(outfile, load_full=False, cube_start = (1982, 1, 1)):
    """
    Temporal distribution of MHW events

    Args:
        outfile:
        mhw_events:
        cube:

    Returns:

    """
    # Standard cube
    std_cube_file = os.path.join(os.getenv('MHW'), 'db', 'MHWevent_cube.nc')
    std_cube = iris.load(std_cube_file)[0]

    # Vary
    vary_cube_file = os.path.join(os.getenv('MHW'), 'db', 'MHWevent_cube_vary.nc')
    vary_cube = iris.load(vary_cube_file)[0]

    if load_full: # Requires >30Gb RAM
        _ = std_cube.data[:]
        _ = vary_cube.data[:]
        print("Cubes fully loaded")

    # Loop on years
    years = 1983 + np.arange(37)
    std_Nevent_year = []
    vary_Nevent_year = []

    for year in years:
        print("year={}".format(year))
        use_iris=False
        if use_iris:
            # Constraint
            time1 = iris.time.PartialDateTime(day=1, year=year, month=1)
            time2 = iris.time.PartialDateTime(day=31, year=year, month=12)
            constraint = iris.Constraint(time=lambda cell: time1 <= cell <= time2)
            # Extract
            std_cube_year = std_cube.extract(constraint)
            vary_cube_year = vary_cube.extract(constraint)
        else:
            t0 = datetime.date(cube_start[0], cube_start[1], cube_start[2]).toordinal()
            ts = datetime.date(year, 1, 1).toordinal()
            te = datetime.date(year, 12, 31).toordinal()
            i0 = ts - t0
            i1 = te - t0 + 1  # Make it inclusive
            # Extract
            std_cube_year = std_cube.data[:,:,i0:i1]
            vary_cube_year = vary_cube.data[:,:,i0:i1]

        #
        std_Nevent_year.append(np.sum(std_cube_year.data[:] >= 1))
        vary_Nevent_year.append(np.sum(vary_cube_year.data[:] >= 1))

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    ax.plot(years, std_Nevent_year, color='k', label='Original')
    ax.plot(years, vary_Nevent_year, color='b', label='Vary')

    # Labels
    ax.set_ylabel(r'Total $N_{\rm Vox}$')
    ax.set_xlabel('Year')
    ax.set_yscale('log')
    ax.minorticks_on()

    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_pacific_blob(outfile):

    # Load MHW Systems
    mhw_sys = pandas.read_hdf(os.path.join(mhw_path,  'MHW_systems_2000.hdf'))#, 'MHW_Events')

    # Pick out the MHW system (TO BE REPLACED BY THE BLOB)
    pacific = (mhw_sys['lat'] > -30.) & (mhw_sys['lat'] < 40.) & (
            mhw_sys['lon'] > 150.) & (mhw_sys['lon'] < 220.)
    extreme = (mhw_sys['NSpax'] > 1e5)
    options = pacific & extreme
    imax = np.argmax(mhw_sys['NSpax'][options].values)
    sys_Id = mhw_sys.index[options][imax]

    blob = mhw_sys.loc[sys_Id]
    print(blob)

    # Load NOAA
    sst = sst_io.load_noaa((blob.date.day, blob.date.month, blob.date.year),
                           subtract_seasonal=True)

    # Cut down to 2x region
    blob_dlat = 0.25*(blob.xboxmax-blob.xboxmin)  # deg
    blob_dlon = 0.25*(blob.yboxmax-blob.yboxmin)  # deg

    # Iris constraints
    sst_frac = 0.55
    latlon_constraint = iris.Constraint(
        latitude=lambda cell: blob.lat-blob_dlat*sst_frac < cell <
                              blob.lat + blob_dlat*sst_frac,
        longitude=lambda cell: blob.lon-blob_dlon*sst_frac < cell <
                               blob.lon + blob_dlon*sst_frac)
    blob_slice = sst.extract(latlon_constraint)

    #embed(header='120 of figs')

    fig = plt.figure(figsize=(7, 5))
    plt.clf()

    # Pacific events
    # Draw the contour with 25 levels.
    #cm = plt.get_cmap('hot')

    proj = ccrs.PlateCarree(central_longitude=-180.0)
    ax = plt.gca(projection=proj)

    qplt.contourf(blob_slice, title='')#, cmap=cm)  # , vmin=0, vmax=20)#, 5)

    # Gridlines
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5,
                      linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    gl.xlocator = mticker.FixedLocator([-180, -140, -100, -60, -20.])

    # Blob
    ax.scatter([blob.lon], [blob.lat], marker='x', color='r',
               transform=ccrs.PlateCarree())
    x0, y0 = blob.lon - blob_dlon/2., blob.lat-blob_dlat/2.
    x1, y1 = blob.lon - blob_dlon/2., blob.lat+blob_dlat/2.
    x2, y2 = blob.lon + blob_dlon/2., blob.lat+blob_dlat/2.
    x3, y3 = blob.lon + blob_dlon/2., blob.lat-blob_dlat/2.
    ax.plot([x0,x1,x2,x3,x0], [y0,y1,y2,y3,y0], color='r', transform=ccrs.PlateCarree())

    # Add coastlines to the map created by contourf.
    plt.gca().coastlines()

    # Layout and save
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_NSys_by_year(outfile, mhw_sys_file=None, lbl=None, vary=False):
    """
    NSys by year


    Args:
        outfile:
        mhw_sys_file:
        lbl:

    Returns:

    """

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    # Do the stats
    years = 1983 + np.arange(36)

    #
    all_Nsys = np.zeros_like(years)
    small_Nsys = np.zeros_like(years)
    int_Nsys = np.zeros_like(years)
    ex_Nsys = np.zeros_like(years)

    #small = mhw_sys.NSpax < 1e3
    #intermediate = (mhw_sys.NSpax >= 1e3) & (mhw_sys.NSpax < 1e5)
    #extreme = (mhw_sys.NSpax >= 1e5)
    small, intermediate, extreme = analy_utils.cut_by_type(mhw_sys)

    for jj, year in enumerate(years):
        # Grab the entries
        in_year = (mhw_sys.date > date(year,1,1)) & (mhw_sys.date < date(year+1,1,1))
        # Total
        all_Nsys[jj] = np.sum(in_year)

        # Cut em up
        small_Nsys[jj] = np.sum(in_year & small)
        int_Nsys[jj] = np.sum(in_year & intermediate)
        ex_Nsys[jj] = np.sum(in_year & extreme)

    #embed(header='3432 of figs_mhwsys')
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])
    #ax_tot.plot(years, all_Nspax, color='k', label='Full')
    ax_tot.plot(years, small_Nsys, color='b', label='Small')
    ax_tot.plot(years, int_Nsys, color='g', label='Intermediate')
    ax_tot.plot(years, ex_Nsys, color='r', label='Extreme')

    ax_tot.set_ylabel(r'$N_{\rm systems}$')
    ax_tot.set_yscale('log')
    ax_tot.set_xlabel('Year')
    ax_tot.xaxis.set_major_locator(plt.MultipleLocator(5.))

    legend = plt.legend(loc='lower left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='x-large', numpoints=1)

    # Font
    set_fontsize(ax_tot, 22.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_Nvox_by_year(outfile, normalize=True, save=True,
                     use_km=True):
    """
    Nvox by year

    Separated by type and method


    Args:
        outfile:
        mhw_sys_file:
        lbl:

    Returns:

    """
    #sys_files = ['MHW_systems.csv', 'MHW_systems_vary.csv',
    #             'MHW_systems_vary_95.csv', 'MCS_systems.csv']
    #labels = ['Standard', 'Vary', 'Vary_95', 'ColdSpells']
    #sys_files = ['MHWS_2019_local.csv', 'MHWS_2019.csv']
    sys_files = ['MHWS_2019.csv', 'MHWS_defaults.csv']
    #sys_files = ['MHWS_2019_local.csv', 'MHWS_defaults.csv']
    #labels = [r'$T_{90}^a$', r'$T_{90}$'] 
    #labels = ['trended', 'standard']
    labels = ['2019', 'Hobday']
    #
    # Stats and normalization
    vary_90_file = os.path.join(noaa_path, 
                                'NOAA_OI_varyclimate_1983-2019.nc')
    clim = xarray.open_dataset(vary_90_file)
    Tthresh_day = clim.threshT.values[0,...]

    R_Earth = 6371. # km
    Area_Earth = 4 * np.pi * R_Earth**2


    # Add up the ocean
    ocean = Tthresh_day != 0
    N_T_day = np.sum(ocean) # Number of cells
    cell_lat = analy_utils.cell_area_by_lat(cell_deg=0.25)

    grid_area = np.outer(cell_lat, np.ones(Tthresh_day.shape[1]))
    ocean_area = np.sum(grid_area[ocean]) # km^2

    print(f"The ocean occupies {N_T_day} cells of [0.25deg]^2")
    print("The ocean is {} of the total gridded area".format(N_T_day/Tthresh_day.size))
    print("The ocean is {} of the total surface area".format(ocean_area/Area_Earth))
    print("There are 10**{} vox in 1 day".format(np.log10(N_T_day)))
    print("There are 10**{} vox in 1 year".format(np.log10(N_T_day*365)))

    if normalize:
        if not use_km:
            norm = N_T_day*365
        else:
            norm = ocean_area*365
    else: 
        norm=1.

    #embed(header='3432 of figs_mhwsys')
    fig = plt.figure(figsize=(8,12))
    gs = gridspec.GridSpec(2,1)
    #fig = plt.figure(figsize=(12, 8))
    #gs = gridspec.GridSpec(2,2)

    for ss, sys_file in zip(np.arange(len(sys_files)), sys_files):
        if ss > 0:
            continue
        # Load MHW Systems
        mhw_sys = mhw_sys_io.load_systems(
            mhw_sys_file=os.path.join(mhw_path, sys_file))
        # Grab Nvox by year
        pd_nvox = analy_utils.Nvox_by_year(mhw_sys, use_km=use_km)

        # Total NSpax
        ax_tot = plt.subplot(gs[ss])

        clrs = ['b', 'g', 'r']
        for key, clr in zip(['random', 'normal', 'extreme'], clrs):
            ax_tot.scatter(pd_nvox.index, getattr(pd_nvox, key)/norm, 
                           color=clr, label=key)
        #ax_tot.scatter(pd_nvox.index, pd_nvox.intermediate, color='g', label=
        #ax_tot.scatter(pd_nvox.index, pd_nvox.extreme, color='r', label='Extreme')

        if normalize:
            ax_tot.set_ylabel(r'Normalized $N_{\rm vox}$ per year')
        else:
            ax_tot.set_ylabel(r'$N_{\rm vox}$ per year')
        #ax_tot.set_yscale('log')
        ax_tot.set_xlabel('Year')
        #ax_tot.set_ylim(5e5, 5e7)
        if use_km:
            ax_tot.set_ylim(0, 0.1)
        else:
            ax_tot.set_ylim(0, 3.5e7/norm)
        ax_tot.xaxis.set_major_locator(plt.MultipleLocator(5.))

        # Label lat, lon
        ax_tot.text(0.4, 0.9, labels[ss], color='black',
                transform=ax_tot.transAxes, ha='center', fontsize=19.)

        # Font
        set_fontsize(ax_tot, 17.)

        if ss == 1:
            legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='x-large', numpoints=1)
        
        # Save?
        if save:
            prs = sys_file.split('.')
            tblfile = prs[0]+'_Vox_by_year.csv'
            # Write
            pd_nvox.to_csv(tblfile)
            print("Wrote: {}".format(tblfile))
            


    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


'''
def fig_NVox_by_year(outfile, mhw_sys_file=None, lbl=None, vary=False):
    """
    NVox by year


    Args:
        outfile:
        mhw_sys_file:
        lbl:

    Returns:

    """

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    # Do the stats
    years = 1983 + np.arange(36)
    # Full Ocean
    all_Nspax = np.zeros_like(years)
    med_Nspax = np.zeros_like(years)

    # Pacific
    pacific_all_Nspax = np.zeros_like(years)
    pacific_med_Nspax = np.zeros_like(years)
    in_pacific = np.zeros_like(mhw_sys.lat, dtype=bool)
    for kk, constraint in enumerate(opy_defs.basin_coords['pacific']):
        in_pacific |= (mhw_sys.lat >= constraint[2]) & (mhw_sys.lat < constraint[3]) & (
                mhw_sys.lon >= constraint[0]) & (mhw_sys.lon < constraint[1])

    # Atlantic
    atlantic_all_Nspax = np.zeros_like(years)
    atlantic_med_Nspax = np.zeros_like(years)
    in_atlantic = np.zeros_like(mhw_sys.lat, dtype=bool)
    for kk, constraint in enumerate(opy_defs.basin_coords['atlantic']):
        in_atlantic |= (mhw_sys.lat >= constraint[2]) & (mhw_sys.lat < constraint[3]) & (
                mhw_sys.lon >= constraint[0]) & (mhw_sys.lon < constraint[1])
    # Indian
    indian_all_Nspax = np.zeros_like(years)
    indian_med_Nspax = np.zeros_like(years)
    in_indian = np.zeros_like(mhw_sys.lat, dtype=bool)
    for kk, constraint in enumerate(opy_defs.basin_coords['indian']):
        in_indian |= (mhw_sys.lat >= constraint[2]) & (mhw_sys.lat < constraint[3]) & (
                    mhw_sys.lon >= constraint[0]) & (mhw_sys.lon < constraint[1])

    for jj, year in enumerate(years):
        # Grab the entries
        in_year = (mhw_sys.date > date(year,1,1)) & (mhw_sys.date < date(year+1,1,1))
        # Total
        all_Nspax[jj] = np.sum(mhw_sys.NSpax[in_year])
        med_Nspax[jj] = np.median(mhw_sys.NSpax[in_year])

        # Pacific
        pacific_all_Nspax[jj] = np.sum(mhw_sys.NSpax[in_year & in_pacific])
        pacific_med_Nspax[jj] = np.median(mhw_sys.NSpax[in_year & in_pacific])

        # Atlantic
        atlantic_all_Nspax[jj] = np.sum(mhw_sys.NSpax[in_year & in_atlantic])
        atlantic_med_Nspax[jj] = np.median(mhw_sys.NSpax[in_year & in_atlantic])

        # Indian
        indian_all_Nspax[jj] = np.sum(mhw_sys.NSpax[in_year & in_indian])
        indian_med_Nspax[jj] = np.median(mhw_sys.NSpax[in_year & in_indian])

    #embed(header='3432 of figs_mhwsys')
    fig = plt.figure(figsize=(8, 12))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])
    ax_tot.plot(years, all_Nspax, color='k', label='Full')
    ax_tot.plot(years, pacific_all_Nspax, color='b', label='Pacific')
    ax_tot.plot(years, atlantic_all_Nspax, color='g', label='Atlantic')
    ax_tot.plot(years, indian_all_Nspax, color='r', label='Indian')

    ax_tot.set_ylabel(r'Total $N_{\rm Vox}$')
    ax_tot.get_xaxis().set_ticks([])
    ax_tot.set_yscale('log')
    ax_tot.minorticks_on()


    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Median NSpax
    ax_med = plt.subplot(gs[1])
    ax_med.plot(years, med_Nspax, color='k')
    ax_med.plot(years, pacific_med_Nspax, color='b')
    ax_med.plot(years, atlantic_med_Nspax, color='g')
    ax_med.plot(years, indian_med_Nspax, color='r')
    ax_med.minorticks_on()

    ax_med.set_ylabel(r'Median $N_{\rm Vox}$')
    ax_med.set_xlabel('Year')
    ax_med.xaxis.set_major_locator(plt.MultipleLocator(5.))

    if lbl is not None:
        ax_med.text(0.9, 0.9, lbl, color='black',
                    transform=ax_med.transAxes, ha='right', fontsize=19.)

    # Font
    for ax in [ax_tot, ax_med]:
        set_fontsize(ax, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))
'''


def fig_NVox_hist(outfile, mhw_sys_file=None, vary=True):

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    # Categories
    cats = np.arange(1,5)
    clrs = ['black', 'blue', 'green', 'red']

    # Bins
    bins = 1. + np.arange(32)*0.25
    if np.log10(np.max(mhw_sys.NVox.values)+1) > bins[-1]:
        bins[-1] = np.log10(np.max(mhw_sys.NVox.values)+1) # Extend to incluce the last one

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    for cat, clr in zip(cats, clrs):
        idx = mhw_sys.category == cat
        # Hist me
        H, bins = np.histogram(np.log10(mhw_sys.NVox[idx].values), bins=bins)
        bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
        plt.step(bincentres, H, where='mid', color=clr, label='Cat={}'.format(cat))

    ax_tot.set_xlabel(r'$\log \, N_{\rm Vox}$')
    ax_tot.set_ylabel('Number of Systems')
    ax_tot.set_yscale('log')
    ax_tot.set_ylim(0.5, 1e6)
    ax_tot.minorticks_on()


    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_duration_hist(outfile, mhw_sys_file=None, vary=True):

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=True)
    mhw_sys['duration'] = mhw_sys.zboxmax - mhw_sys.zboxmin + 1

    # Categories
    cats = np.arange(1,5)
    clrs = ['black', 'blue', 'green', 'red']

    # Bins
    bins = np.linspace(1., 3.5, 10)
    bins[-1] = np.log10(np.max(mhw_sys.duration.values)+1)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    for cat, clr in zip(cats, clrs):
        idx = mhw_sys.category == cat
        # Hist me
        H, bins = np.histogram(np.log10(mhw_sys.duration[idx].values), bins=bins)
        bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
        plt.step(bincentres, H, where='mid', color=clr, label='Cat={}'.format(cat))

    ax_tot.set_xlabel('Duration (days)')
    ax_tot.set_ylabel('Number of Systems')
    ax_tot.set_yscale('log')
    ax_tot.set_ylim(0.5, 2e5)
    ax_tot.minorticks_on()


    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_maxarea_hist(outfile, mhw_sys_file=None, vary=True):

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    # Categories
    cats = np.arange(1,5)
    clrs = ['black', 'blue', 'green', 'red']

    # Bins
    bins = np.linspace(1., 3.5, 10)
    bins[-1] = np.log10(np.max(mhw_sys.max_area.values)+1) # Extend to incluce the last one

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    for cat, clr in zip(cats, clrs):
        idx = mhw_sys.category == cat
        # Hist me
        H, bins = np.histogram(np.log10(mhw_sys.max_area[idx].values), bins=bins)
        bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
        plt.step(bincentres, H, where='mid', color=clr, label='Cat={}'.format(cat))

    ax_tot.set_xlabel(r'Maximum Area (0.25deg)$^2$')
    ax_tot.set_ylabel('Number of Systems')
    ax_tot.set_yscale('log')
    ax_tot.set_ylim(10., 1e5)
    ax_tot.minorticks_on()


    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_dur_vs_NVox(outfile, mhw_sys_file=None, vary=True,
                    xcorner=False, seab=False):
    """

    Args:
        outfile:
        mhw_sys_file:
        vary:
        xcorner:

    Returns:

    """

    # A few famous ones
    pacific_blob = dict(lon=360.-145, lat=48., lbl='Pacific Blob', marker='o',
                        date=datetime.date(2014,1,15), mask_Id=32106)
    NW_atlantic = dict(lon=360.-70, lat=40., lbl='Northwest Atlantic', marker='*',
                        date=datetime.date(2011,12,15), mask_Id=723727)
    NAustralia = dict(lon=140., lat=-15., lbl='North Australia', marker='x',
                       date=datetime.date(2016,2,1), mask_Id=93602)
    famous = [pacific_blob, NW_atlantic, NAustralia]

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)
    mhw_sys['duration'] = mhw_sys.zboxmax - mhw_sys.zboxmin + 1

    # Setup
    #lat = -89.875 + np.arange(720)*0.25
    #lon = 0.125 + np.arange(1440)*0.25

    #bad_sys = (mhw_sys.mask_Id == 32106)

    # Find the famous ones
    iloc_famous = []
    for ifamous in famous:
        xlat = int((ifamous['lat'] + 89.875) / 0.25)
        ylon = int((ifamous['lon'] + 0.125) / 0.25)
        zt = ifamous['date'].toordinal() - datetime.date(1982,1,1).toordinal()
        #
        gd_lat = (mhw_sys.xboxmin <= xlat) & (xlat <= mhw_sys.xboxmax)
        gd_lon = (mhw_sys.yboxmin <= ylon) & (ylon <= mhw_sys.yboxmax)
        gd_t = (mhw_sys.zboxmin <= zt) & (zt <= mhw_sys.zboxmax)
        if 'mask_Id' in ifamous.keys():
            gd_mask = mhw_sys.mask_Id == ifamous['mask_Id']
        else:
            gd_mask = np.ones(gd_lat.size, dtype=bool)
        all_gd = gd_lon & gd_lat & gd_t & gd_mask #& np.logical_not(bad_sys)
        #
        if np.sum(all_gd) > 1:
            print("Multiple for {}".format(ifamous))
            embed(header='667')
        elif np.sum(all_gd) == 0:
            print("None for {}".format(ifamous))

        iloc_famous.append(np.where(all_gd)[0][0])

    # Bins
    bins_NVox = 1. + np.arange(32)*0.25
    if np.log10(np.max(mhw_sys.NVox.values)+1) > bins_NVox[-1]:
        bins_NVox[-1] = np.log10(np.max(mhw_sys.NVox.values)+1)  # Extend to incluce the last one
    bins_dur = np.linspace(0.5, 4.0, 35)
    #if np.max(mhw_sys.duration.values) > bins_dur[-1]:
    #    bins_dur[-1] = np.log10(np.max(mhw_sys.duration.values)+1)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NVox
    ax_tot = plt.subplot(gs[0])

    if xcorner:
        # 2D hist
        hist2d(np.log10(mhw_sys.NVox.values), np.log10(mhw_sys.duration.values),
               bins=[bins_NVox, bins_dur], ax=ax_tot, color='b')
    elif seab:
        sns.histplot(data=mhw_sys, x='duration', y='NVox',
                     log_scale=True, ax=ax_tot, bins=30,
                     cbar=True, pthresh=.05, pmax=.9)
    else:
        counts, xedges, yedges = np.histogram2d(np.log10(mhw_sys.duration.values),
            np.log10(mhw_sys.NVox.values), bins=(bins_dur, bins_NVox))
        cm = plt.get_cmap('Blues')
        mplt = ax_tot.pcolormesh(xedges,yedges,np.log10(counts.transpose()), cmap=cm)
        cb = plt.colorbar(mplt, fraction=0.030, pad=0.04)
        cb.ax.tick_params(labelsize=13.)
        cb.set_label(r'log10 Counts', fontsize=20.)

    # Add famous ones
    for kk, iloc in enumerate(iloc_famous):
        imhw_sys = mhw_sys.iloc[iloc]
        ax_tot.plot(np.log10([imhw_sys.duration]), np.log10([imhw_sys.NVox]), color='r',
                       label=famous[kk]['lbl'], marker=famous[kk]['marker'],
                    linestyle='None')

    #ax_tot.set_yscale('log')
    if not seab:
        ax_tot.set_xlim(0.5, 4.0)
        ax_tot.set_ylabel(r'$\log_{10} \, N_{\rm Vox} \; (vox)$')
        ax_tot.set_xlabel(r'$\log_{10} \, t_{\rm dur}$ (days)')
    #ax_tot.minorticks_on()

    legend = plt.legend(loc='lower right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 21.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    # Stats
    extreme = mhw_sys.NVox > defs.type_dict['extreme'][0]
    print("Minimum duration for extreme {}".format(
        mhw_sys[extreme].duration.min()))


def fig_maxA_vs_NVox(outfile, mhw_sys_file=None, vary=True):

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, vary=vary)

    # Bins
    bins_NVox = 1. + np.arange(32)*0.25
    if np.log10(np.max(mhw_sys.NSpax.values)+1) > bins_NVox[-1]:
        bins_NVox[-1] = np.log10(np.max(mhw_sys.NSpax.values)+1)  # Extend to incluce the last one
    bins_maxA = np.linspace(0.5, 3.5, 40)
    if np.max(mhw_sys.max_area.values) > bins_maxA[-1]:
        bins_maxA[-1] = np.log10(np.max(mhw_sys.max_area.values)+1)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    # 2D hist
    hist2d(np.log10(mhw_sys.NSpax.values), np.log10(mhw_sys.max_area.values),
           bins=[bins_NVox, bins_maxA], ax=ax_tot, color='g')

    ax_tot.set_xlabel('log10 NVox')
    ax_tot.set_ylabel('log10 MaxArea')
    ax_tot.set_ylim(0.3, 5.0)
    #ax_tot.minorticks_on()

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                    handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_climate(outfile):

    # Load up climates
    scale_file_2012 = os.path.join(resource_filename('mhw', 'data'), 'climate',
                                   'noaa_median_climate_1983_2012.hdf')
    scale_file_2019 = os.path.join(resource_filename('mhw', 'data'), 'climate',
                              'noaa_median_climate_1983_2019.hdf')
    #scale_tbl_2012 = pandas.read_hdf(scale_file_2012)
    scale_tbl_2019 = pandas.read_hdf(scale_file_2019)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    # Median dSST
    dates = mpl.dates.date2num(scale_tbl_2019.index)
    ax.plot_date(dates, scale_tbl_2019.medSSTa, 'o', ms=1, label='Measured')
    ax.plot_date(dates, scale_tbl_2019.medSSTa_savgol, 'r-', label='Savgol Filtered',
                 lw=2.5)

    ax.set_xlabel('Year')
    ax.set_ylabel(r'Median $\Delta$SST (K)')
    #ax_tot.set_yscale('log')
    #ax_tot.set_ylim(10., 1e5)
    #ax_tot.minorticks_on()

    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax, 21.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_T_thresh(outfile, lat=36.125, lon=220.125):

    # Load up
    hobday_90_file = os.path.join(os.getenv('NOAA_OI'), 
                                  'NOAA_OI_climate_1983-2012.nc')
    full_90_file = os.path.join(os.getenv('NOAA_OI'), 
                                'NOAA_OI_climate_1983-2019.nc')
    local_90_file = os.path.join(os.getenv('NOAA_OI'), 
                                'NOAA_OI_detrend_local_climate_1983-2019.nc')
    global_90_file = os.path.join(os.getenv('NOAA_OI'), 
                                'NOAA_OI_detrend_median_climate_1983-2019.nc')
    #vary_95_file = os.path.join(os.getenv('NOAA_OI'), 
    #                            'NOAA_OI_detrend_local_climate_1983-2019.nc')

    hobday_90 = xarray.open_dataset(hobday_90_file)
    full_90 = xarray.open_dataset(full_90_file)
    local_90 = xarray.open_dataset(local_90_file)
    global_90 = xarray.open_dataset(global_90_file)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Loop me
    #labels = ['Hobday', '1983-2019', 'Vary 90th', 'Vary 95th']
    clrs = ['k', 'b', 'r', 'orange']
    labels = [r'$T_{90}^{2012}$', r'$T_{90}^{2019}$', 
              r'$T_{90}^{\rm local}$', r'$T_{90}^{\rm global}$']
    for ss, lbl, ds, clr in zip(np.arange(len(labels)), labels,
                       [hobday_90, full_90, local_90, global_90],
                       clrs):

        dkey = 'day' if 'day' in ds.coords.keys() else 'doy'
        da = ds.sel(lat=lat, lon=lon)
        # Plot
        ax0.plot(ds[dkey].data,  da.threshT.data, label=lbl,
                 color=clr)

        # Residuals off of Hobday
        if ss == 0:
            Hobday_threshT = da.threshT.data
        else:
            ax1.plot(ds[dkey].data,  
                     da.threshT.data-Hobday_threshT, label=lbl,
                     color=clr)

    # Label lat, lon
    ax0.text(0.9, 0.1, 'lat={}deg\n lon={}deg'.format(lat,lon), color='black',
                    transform=ax0.transAxes, ha='right', fontsize=19.)

    ax1.set_xlabel('DOY')
    ax0.set_ylabel(r'$T_{90}$ (deg C)')
    ax1.set_ylabel(r'$\Delta T_{90}$ (deg C)')
    #ax_tot.set_yscale('log')
    #ax_tot.set_ylim(10., 1e5)
    #ax_tot.minorticks_on()

    ax0.axes.xaxis.set_ticklabels([])


    for ax in [ax0, ax1]:
        set_fontsize(ax, 21.)
        ax.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_SST_vs_Tthresh(outfile, lat=36.125, lon=220.125):

    # Load up
    #hobday_90_file = os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2012.nc')
    #full_90_file = os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019.nc')
    vary_90_file = os.path.join(noaa_path, 'NOAA_OI_varyclimate_1983-2019.nc')
    scale_file = 's3://mhw/climate/noaa_median_climate_1983_2019.parquet'
    scale_df = ulmo_io.load_main_table(scale_file)
    #vary_95_file = os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_varyclimate_1983-2019_95.nc')

    #hobday_90 = xarray.open_dataset(hobday_90_file)
    #full_90 = xarray.open_dataset(full_90_file)
    vary_90 = xarray.open_dataset(vary_90_file)
    #vary_95 = xarray.open_dataset(vary_95_file)

    # SST for 2016
    year = 2015
    noaa_file = os.path.join(noaa_path, 'sst.day.mean.{}.nc'.format(year))
    sst_ds = xarray.open_dataset(noaa_file)
    sst_da = sst_ds.sel(lat=lat, lon=lon)

    # Prep for scaling
    scale_dates = (scale_df.index >= sst_da.time.data[0]) & (
        scale_df.index <= sst_da.time.data[-1]) 

    # Figure
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)


    # Loop me
    dkey = 'day' if 'day' in vary_90.coords.keys() else 'doy'
    da = vary_90.sel(lat=lat, lon=lon)

    for ss in range(2):
        ax = plt.subplot(gs[ss])
        # Plot
        ax.plot(vary_90[dkey].data,  da.threshT.data, label='T90', color='k')

        # SST
        if ss == 0:
            ax.scatter(np.arange(sst_da.sst.size), sst_da.sst.data, 
                   label='SST {}'.format(year), color='b')
        else:
            ax.scatter(np.arange(sst_da.sst.size), 
                       sst_da.sst.data-scale_df[scale_dates].medSSTa_savgol, 
                   label="SST' {}".format(year), color='r')

        # Label lat, lon
        ax.text(0.9, 0.1, 'lat={}, lon={}'.format(lat,lon), color='black',
                    transform=ax.transAxes, ha='right', fontsize=19.)

        ax.set_xlabel('DOY')
        ax.set_ylabel(r'$T_{\rm thresh}$ (deg C)')

    #ax_tot.set_yscale('log')
    #ax_tot.set_ylim(10., 1e5)
    #ax_tot.minorticks_on()

        legend = ax.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=19, numpoints=1)

        set_fontsize(ax, 21.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_location_NVox(ext, size, vary=True, nside=64, nmax=10):

    # Outfile
    fig_root = 'fig_location_NVox'
    outfile = fig_root+'_{}.{}'.format(size, ext)

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(vary=vary)
    print("N MHW Systems = {}".format(len(mhw_sys)))

    # Cut on size
    if size == 'low':
        sub_sys = mhw_sys[mhw_sys.NSpax <= 1e3]
    elif size == 'int':
        sub_sys = mhw_sys[(mhw_sys.NSpax <= 1e6) & (mhw_sys.NSpax > 1e3)]
    else:
        embed(header='678 of figs')

    # Healpix me
    hp_events = mhw_sys_utils.mhw_sys_to_healpix(sub_sys, nside)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    # Median dSST
    hp.mollview(hp_events, min=1, max=nmax,
                flip='geo', title='', unit=r'$N_{\rm sys}$',
                rot=(0., 180., 180.))

    # Layout and save
    #plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_example_mhws(outfile, mhw_sys_file=os.path.join(
                            os.getenv('MHW'), 'db', 'MHWS_2019.csv'),
                        vary=False,
                     #mask_Id=1575120, 
                     make_mayavi=False,
                     mask_Id=1458524): # 2019
                     #mask_Id=544711, # 2019
                     #mask_Id=1475052): # 2019, local
    #1489500):

    mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_mask.nc')
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=vary)


    # Find a sys
    if False:
        gd_dur = mhw_sys.duration > 120.
        gd_lat = (mhw_sys.lat > 0.) & (mhw_sys.lat < 50.)
        gd_lon = (mhw_sys.lon > 190.) & (mhw_sys.lon < 250.)
        all_gd = gd_dur & gd_lat & gd_lon
        mhw_sys[all_gd]
    idx = np.where(mhw_sys.mask_Id == mask_Id)[0][0]
    isys = mhw_sys.iloc[idx]
    sys_startdate = isys.datetime - datetime.timedelta(days=int(isys.zcen)-int(isys.zboxmin))


    # Grab the mask
    mask_da = mhw_sys_io.load_mask_from_system(isys, vary=vary,
                                               mhw_mask_file=mask_file)

    # Patch
    fov = 70.  # deg
    lat_min = isys.lat - fov/2.
    lat_max = isys.lat + fov/2.
    lon_min = isys.lon - fov/2.
    lon_max = isys.lon + fov/2.

    # Load up climates
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(3,2)

    #toffs = [140, 40, 10]  # days
    toffs = [10, 40, 200]  # days

    for ss, toff in enumerate(toffs):
        ax = plt.subplot(gs[ss,0])#, projection=proj)

        # Grab SST
        off_date = sys_startdate + datetime.timedelta(days=toff)
        print(f"The date is {off_date}")
        sst = sst_io.load_noaa((off_date.day, off_date.month, off_date.year),
                               subtract_seasonal=True)
        # Slice
        sst_slice = sst.sel(lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max))
        sst_img = sst_slice.data[:]

        # Slice the mask too
        mask_cut = mask_da.sel(time=off_date, lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max))
        mask = mask_cut.data[:]

        region = mask == mask_Id

        # https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html

        #ax.imshow(sst_img*region, origin='lower')
        ax.imshow(sst_img, origin='lower')

        # Outline
        #https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
        segments = mk_segments(region, region.shape[1], region.shape[0])
        ax.plot(segments[:, 0], segments[:, 1], color='k', linewidth=0.5)

        # Axes
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax.text(0.05, 0.05, f"+{toff} days", color='black',
                transform=ax.transAxes, ha='left', fontsize=19.)

        # Add coastlines to the map created by contourf.
        #plt.gca().coastlines()

        # Turn off Title
        #ax.title('')

        # View
        #ax.set_xlabel('Year')
        #ax.set_ylabel(r'Median $\Delta$SST (K)')
    #ax_tot.set_yscale('log')
    #ax_tot.set_ylim(10., 1e5)
    #ax_tot.minorticks_on()

    set_fontsize(ax, 19.)

    # Mayavi
    if make_mayavi:
        from mayavi import mlab
        mask_cut = mask_da.sel(lon=slice(lon_min, lon_max),
                               lat=slice(lat_min, lat_max)).data[:]
        # Zero out
        mask_cut[mask_cut != mask_Id] = 0
        scale = 1.
        #visualize.test_mayavi(mask_cut, scale=1.)

        cntrs = np.unique(mask_cut)[1:]
        size = (1050, 1050)
        fig = mlab.figure(1, bgcolor=(0.0, 0.0, 0.0),
                          size=size)

        mlab.contour3d(mask_cut, transparent=True,
                           contours=cntrs.tolist(),
                           opacity=0.7, vmin=1)

        # Axes
        mlab.axes(color=(1, 1, 1),
                  extent=[0., scale * mask_cut.shape[0], 0.,
                          scale * mask_cut.shape[1], 1, mask_cut.shape[2]],
                  xlabel='lon', ylabel='lat', zlabel='day')
        ifig = mlab.gcf()
        png_file = 'tmp.png'
        mlab.savefig(filename=png_file, figure=ifig)

        # Read from disk
        ax_img = plt.subplot(gs[:, 1:])  # , projection=proj)
        img = mpimg.imread(png_file)
        ax_img.imshow(img)

        ax_img.get_xaxis().set_ticks([])
        ax_img.get_yaxis().set_ticks([])

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def fig_int_gallery(sys_file, mask_file, outfile, rand_dict, debug=False):
    """
    Gallery of Intermediate MHWS

    Args:
        sys_file:
        mask_file:
        outfile:
        rand_dict:
        debug:

    Returns:

    """
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=os.path.join(mhw_path, sys_file))

    if debug:
        mask_Id=1575120  # Only for vary
        idx = np.where(mhw_sys.mask_Id == mask_Id)[0][0]
        isys = mhw_sys.iloc[idx]

    climate_file = os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019.nc')

    rand_sys = analy_utils.random_sys(mhw_sys, np.arange(1984,2019,4), 4, **rand_dict)

    # Load up climates
    fig = plt.figure(figsize=(9, 9))
    plt.clf()
    gs = gridspec.GridSpec(3,3)

    for ss in range(9):
        isys = rand_sys.iloc[ss]
        # Patch
        fov = 30.  # deg
        lat_min = isys.lat - fov/2.
        lat_max = isys.lat + fov/2.
        lon_min = isys.lon - fov/2.
        lon_max = isys.lon + fov/2.

        ax = plt.subplot(gs[ss])#, projection=proj)

        # Grab SST
        sst = sst_io.load_noaa((isys.datetime.day, isys.datetime.month, isys.datetime.year),
                               subtract_seasonal=True, climate_file=climate_file)
        # Slice
        sst_slice = sst.sel(lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max))
        sst_img = sst_slice.data[:]

        # Grab the mask
        mask_da = mhw_sys_io.load_mask_from_system(isys, mhw_mask_file=os.path.join(mhw_path, mask_file))

        # Slice
        mask_cut = mask_da.sel(time=slice(isys.startdate, isys.enddate), lon=slice(lon_min, lon_max),
                               lat=slice(lat_min, lat_max))
        mask = mask_cut.data[:]
        vox = (mask == isys.mask_Id).astype(int)
        vox_smash = np.sum(vox, axis=2)


        # https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html

        #ax.imshow(sst_img*region, origin='lower')
        ax.imshow(sst_img, origin='lower', cmap='jet')

        # Outline
        #https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
        region = vox_smash > 0
        segments = mk_segments(region, region.shape[1], region.shape[0])
        ax.plot(segments[:, 0], segments[:, 1], color='k', linewidth=0.5)

        # Axes
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Label lat, lon
        ax.text(0.1, 0.92, 'lat={:0.2f}, lon={:0.2f}'.format(isys.lat, isys.lon),
                color='black', transform=ax.transAxes, ha='left', fontsize=13.)
        ax.text(0.1, 0.82, '{}'.format(isys.date),
                color='black', transform=ax.transAxes, ha='left', fontsize=13.)

        # Add coastlines to the map created by contourf.
        #plt.gca().coastlines()

        # Turn off Title
        #ax.title('')

        # View
        #ax.set_xlabel('Year')
        #ax.set_ylabel(r'Median $\Delta$SST (K)')
    #ax_tot.set_yscale('log')
    #ax_tot.set_ylim(10., 1e5)
    #ax_tot.minorticks_on()

    set_fontsize(ax, 19.)


    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_extreme_gallery(sys_file, mask_file, outroot, next=10, debug=False,
                        cold=False):
    """
    Page by page 'gallery' of Extreme MHWS
    
    https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html

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
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=os.path.join(mhw_path, sys_file))

    # Grab the extreme systems
    isrt = np.argsort(mhw_sys.NVox)

    if debug:
        ext_sys = mhw_sys.iloc[isrt[660500:660500 + next]]
        next = 1
    else:
        ext_sys = mhw_sys.iloc[np.flip(isrt[-1*next:])]

    cbfsz = 15.

    for kk in range(next):
        isys = ext_sys.iloc[kk]

        proj = ccrs.PlateCarree(central_longitude=isys.lon)

        fig = plt.figure(figsize=(6, 7))
        gs = gridspec.GridSpec(2, 1)

        # Duration sub-plot
        ax = plt.subplot(gs[0], projection=proj)
        ax.coastlines()

        # Grab the mask
        mask_da = mhw_sys_io.load_mask_from_system(
            isys, mhw_mask_file=os.path.join(mhw_path, mask_file))

        # Slice
        mask_cut = mask_da.sel(time=slice(isys.startdate, isys.enddate))
                               #lon=slice(lon_min, lon_max),
                               #lat=slice(lat_min, lat_max))
        mask = mask_cut.data[:]


        # Duration
        vox = (mask == isys.mask_Id).astype(int)
        vox_smash = np.sum(vox, axis=2).astype(float)
        vox_smash[vox_smash==0] = np.nan

        if cold:
            cm = plt.get_cmap('Blues')
        else:
            cm = plt.get_cmap('YlOrRd')
        a1 = ax.pcolormesh(mask_da.lon, mask_da.lat, vox_smash, cmap=cm, transform=ccrs.PlateCarree())
        cb = plt.colorbar(a1, fraction=0.030, pad=0.04)
        cb.set_label('Duration (days)', fontsize=cbfsz)

        add_gridlines(ax)

        # Label lat, lon
        #ax.text(0.1, 0.92, 'lat={:0.2f}, lon={:0.2f}'.format(isys.lat, isys.lon),
        #        color='black', transform=ax.transAxes, ha='left', fontsize=13.)
        #ax.text(0.1, 0.82, '{}'.format(isys.date),
        #        color='black', transform=ax.transAxes, ha='left', fontsize=13.)

        # Add coastlines to the map created by contourf.
        # plt.gca().coastlines()

        # Title
        plt.title('{} NVox={}'.format(isys.date, isys.NVox))


        # First day
        ivox = np.where(vox)

        bdval = 999999999
        mask[:] = bdval
        mask[ivox] = ivox[2]
        day_onset = np.min(mask, axis=2).astype(float)
        day_onset[day_onset==bdval] = np.nan

        # Duration sub-plot
        ax = plt.subplot(gs[1], projection=proj)
        ax.coastlines()

        #onset_da = xarray.DataArray(day_onset, coords=[mask_da.lat, mask_da.lon])
        #cm = plt.get_cmap('gist_rainbow')
        #cm = plt.get_cmap('plasma')
        if cold:
            cm = plt.get_cmap('autumn')
        else:
            cm = plt.get_cmap('winter')
        a2 = ax.pcolormesh(mask_da.lon, mask_da.lat, day_onset, cmap=cm, transform=ccrs.PlateCarree())
        cb = plt.colorbar(a2, fraction=0.030, pad=0.04)
        cb.set_label('Onset since {:s} (days)'.format(isys.startdate.strftime('%Y-%m-%d')),
                     fontsize=cbfsz)
        #onset_da.plot(ax=ax, transform=proj, cmap=cm,
        #               cbar_kwargs = {'label': 'Onset (days)',
        #                                'fraction': 0.020, 'pad': 0.04})
        add_gridlines(ax)

        set_fontsize(ax, 19.)

        del mask, mask_da, mask_cut

        # Layout and save
        #plt.tight_layout(pad=0.2, h_pad=0., w_pad=0.1)
        outfile = outroot+'_{}.png'.format(kk)
        plt.savefig(outfile, dpi=300)
        plt.close()
        print('Wrote {:s}'.format(outfile))


def fig_extreme_evolution(sys_file, mask_file, ordinal, outfile, 
                          debug=False, cold=False,
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
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=os.path.join(mhw_path, sys_file))

    # Grab the extreme systems
    isrt = np.argsort(mhw_sys.NVox.values)

    # Grab it
    isys = mhw_sys.iloc[isrt[-1-ordinal]]
    tot_days = (isys.enddate -isys.startdate).days

    # Grab the mask
    mask_da = mhw_sys_io.load_mask_from_system(
        isys, mhw_mask_file=os.path.join(mhw_path, mask_file))


    proj = ccrs.PlateCarree(central_longitude=isys.lon)
    cbfsz = 15.

    fig = plt.figure(figsize=(6, 7))
    nrow, ncol = 3,3
    gs = gridspec.GridSpec(nrow, ncol)
    # Duration sub-plot


    #for ss in range(2): 
    for ss in range(nrow*ncol):
        if ss == 0:
            toff = 10
        elif ss == nrow*ncol-1:
            toff = tot_days-10
        else:
            toff = 10 + ss*int(np.round((tot_days-20)/(nrow*ncol-1)))

        ax = plt.subplot(gs[ss], projection=proj)
        ax.coastlines()

        # Slice

        itime = isys.startdate+pandas.Timedelta('{}days'.format(toff))
        print(ss, itime)
        mask_cut = mask_da.sel(
                time=slice(itime-pandas.Timedelta('1days'), itime),
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
        
        if cold:
            cm = plt.get_cmap('Blues')
        else:
            cm = plt.get_cmap('YlOrRd')
        a1 = ax.pcolormesh(sst.lon, sst.lat, sst, 
                        cmap=cm, transform=ccrs.PlateCarree())
        #cb = plt.colorbar(a1, fraction=0.030, pad=0.04)
        #cb.set_label('Duration (days)', fontsize=cbfsz)

        # Set ranges

        #add_gridlines(ax)

    # Label lat, lon
    #ax.text(0.1, 0.92, 'lat={:0.2f}, lon={:0.2f}'.format(isys.lat, isys.lon),
    #        color='black', transform=ax.transAxes, ha='left', fontsize=13.)
    #ax.text(0.1, 0.82, '{}'.format(isys.date),
    #        color='black', transform=ax.transAxes, ha='left', fontsize=13.)

    # Add coastlines to the map created by contourf.
    # plt.gca().coastlines()

    # Title
    #plt.title('{} NVox={}'.format(isys.date, isys.NVox))

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def fig_MHWS_histograms(outfile, 
                        mhw_sys_file=os.path.join(
                            os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
                            show_insets=False,
                        use_km=True):
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file)#, vary=vary)
    mhw_sys['duration'] = mhw_sys.zboxmax - mhw_sys.zboxmin + 1

    # km?
    if use_km:
        vox_key = 'NVox_km'
        area_key = 'max_area_km'
        type_dict = defs.type_dict_km
    else:
        vox_key = 'NVox'
        area_key = 'max_area'
        type_dict = defs.type_dict

    # Trim on duration >=5
    mhw_sys = mhw_sys[mhw_sys.duration >= 5].copy()

    #embed(header='1637 figs')

    # Categories
    cats = np.arange(1,5)
    clrs = ['black', 'blue', 'green', 'red']


    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss in range(4):
        # Bins
        ylabel = 'Number of MHWSs'
        if ss >= 2:
            if use_km:
                bins = 1. + np.arange(38)*0.25
            else:
                bins = 1. + np.arange(32)*0.25
            attr = vox_key
            if np.log10(np.max(mhw_sys[attr].values)+1) > bins[-1]:
                bins[-1] = np.log10(np.max(mhw_sys[attr].values)+1) # Extend to incluce the last one
            xlabel = r'$N_{\rm Vox}$'
            if use_km:
                xlabel += r'$\; (\rm{days \, km^2})$'
            clr = 'b'
            if ss == 3:
                ylabel = r'Fraction of total $N_{\rm Vox}$'
        elif ss == 0:
            bins = np.linspace(np.log10(5), 4.0, 32)
            attr = 'duration'
            if np.log10(np.max(mhw_sys[attr].values)+1) > bins[-1]:
                bins[-1] = np.log10(np.max(mhw_sys[attr].values)+1) # Extend to incluce the last one
            xlabel = r'$t_{\rm dur} \; ({\rm days})$'
            clr = 'g'
        elif ss == 1: # Max area
            if use_km:
                bins = np.linspace(0., 8.0, 32)
            else:
                bins = np.linspace(0., 5.0, 32)
            attr = area_key
            if np.log10(np.max(mhw_sys[attr].values)+1) > bins[-1]:
                bins[-1] = np.log10(np.max(mhw_sys[attr].values)+1) # Extend to incluce the last one
            if use_km:
                xlabel = r'$A_{\rm max} \; ({\rm km^2})$'
            else:
                xlabel = r'$A_{\rm max} \; ({\rm 0.25deg^2})$'
            clr = 'r'
            #ylim = (0.5, 1e6)

        ax = plt.subplot(gs[ss])

        # Hist me
        if ss <= 2:
            sns.histplot(mhw_sys, x=attr, bins=bins, log_scale=True, ax=ax,
                     color=clr)
        else:
            '''
            # Original -- Sum Nvox
            xbin = []
            tot_Vox = []
            for tt in range(len(bins)-1):
                in_bin = (mhw_sys[attr] >= 10**bins[tt]) & (mhw_sys[attr] < 10**bins[tt+1]) 
                #
                xbin.append(np.mean(bins[tt:tt+1]))
                tot_Vox.append(np.sum(mhw_sys[in_bin][attr]))
            ax.stairs(tot_Vox, 10**bins, color='k', fill=True) 

            # Finish
            ax.set_xscale('log')
            xlabel = r'$N_{\rm Vox}$'
            ylabel = 'Total Vox'
            '''
            # Cumulative
            xval = mhw_sys[vox_key].values
            srt = np.argsort(xval)
            xdata = xval[srt]
            CDF_data = np.cumsum(xval[srt]) / np.sum(xval)
            ax.plot(xdata, CDF_data, label='data', color='black',
                    drawstyle='steps-mid')
            ax.set_xscale('log')
            ax.set_ylim(0., 1.05)

            # Label regions
            ax.axvline(type_dict['normal'][0], color='gray', ls='--')
            ax.axvline(type_dict['normal'][1], color='gray', ls='--')

        #for cat, clr in zip(cats, clrs):
        #    idx = mhw_sys.category == cat
        #bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
        #plt.step(bincentres, H, where='mid', color=clr, label='Cat={}'.format(cat))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ss <= 2:
            ax.set_yscale('log')
        #ax.set_ylim(0.5, 1e6)
        ax.minorticks_on()

        # Font size
        set_fontsize(ax, 19.)

        # ###############################################3333
        # Insets
        if show_insets:
            if ss == 0:
                xmin = type_dict['dur_xmin']
                alpha_mnx = (-4., -2.)
            elif ss == 1:
                xmin = type_dict['area_xmin']
                alpha_mnx = (-3.5, -1.01)
            elif ss == 2:
                xmin = type_dict['vox_xmin']
                alpha_mnx = (-3.5, -1.01)
            else: 
                continue

            xval = mhw_sys[mhw_sys[attr] >= xmin][attr].values

            if ss == 0 or (not use_km):
                C, best_alpha, alpha, logL = fitting.fit_discrete_powerlaw(
                    xval, xmin, alpha_mnx)
            else:
                C, best_alpha = fitting.fit_continuous_powerlaw(
                    xval, xmin)

            ax_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
            if ss == 0 or (not use_km):
                val = np.arange(xmin, xval.max()+1)
                dx = 1.
            else:
                val = 10**np.linspace(np.log10(xval.min()), np.log10(xval.max()), 10000)
                dx = val - np.roll(val,1)
                dx[0] = dx[1]

            Pt_fit = C*val**(best_alpha)
            CDF_fit = np.cumsum(Pt_fit*dx)/np.sum(Pt_fit*dx)

            # Data
            if ss == 0 or (not use_km):
                xdata, counts = np.unique(xval, return_counts=True)
                CDF_data = np.cumsum(counts)/np.sum(counts)
            else:
                srt = np.argsort(xval)
                xdata = xval[srt]
                CDF_data = np.arange(xval.size) / (xval.size-1)

            # Plot
            ax_inset.plot(val, CDF_fit, label=r'$\alpha='+'{:0.1f}'.format(
                best_alpha)+'$)', color='gray')
            ax_inset.plot(xdata, CDF_data, label='data', color='orange',
                    drawstyle='steps-mid')
            ax_inset.legend(fontsize=9., loc='lower right')
            ax_inset.set_xscale('log')
            ax_inset.set_ylim(0., 1.05)
            ax_inset.set_xlabel(xlabel)
            ax_inset.set_ylabel('CDF')  


    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                    handletextpad=0.3, fontsize=19, numpoints=1)


    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    # A few extra stats
    extreme = mhw_sys[vox_key] > type_dict['extreme'][0]
    total_NVox = np.sum(mhw_sys[vox_key])
    NVox_extreme = np.sum(mhw_sys[extreme][vox_key])
    print("There are {} extreme MHWS.".format(np.sum(extreme)))
    print("Extreme are {} percent of the total".format(100.*NVox_extreme/total_NVox,))



def fig_cumulative_NVox(outfile, mhw_sys_file=os.path.join(
                            os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
                        use_km=True):
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file)#, vary=vary)
    mhw_sys['duration'] = mhw_sys.zboxmax - mhw_sys.zboxmin + 1

    # km?
    if use_km:
        vox_key = 'NVox_km'
        area_key = 'max_area_km'
        type_dict = defs.type_dict_km
    else:
        vox_key = 'NVox'
        area_key = 'max_area'
        type_dict = defs.type_dict

    # Trim on duration >=5
    mhw_sys = mhw_sys[mhw_sys.duration >= 5].copy()

    # Begin the figure
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    xval = mhw_sys[vox_key].values
    srt = np.argsort(xval)
    xdata = xval[srt]
    CDF_data = np.cumsum(xval[srt]) / np.sum(xval)

    # Plot
    ax.plot(xdata, CDF_data, label='data', color='black',
                drawstyle='steps-mid')
    ax.set_xscale('log')
    ax.set_ylim(0., 1.05)

    xlabel = r'$N_{\rm Vox}$'
    if use_km:
        xlabel += r'$\; (\rm{days \, km^2})$'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')  

    # Font size
    set_fontsize(ax, 19.)

    # Mark regions
    ax.axvline(type_dict['normal'][0], color='gray', ls='--')
    ax.axvline(type_dict['normal'][1], color='gray', ls='--')

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def count_systems(list_of_Ids, mask, spat_systems):
    # Proceed
    for mask_Id in list_of_Ids:
        vox = mask == mask_Id
        cvox = np.any(vox, axis=2).astype(int)
        spat_systems += cvox
    # Return
    return spat_systems


def fig_spatial_systems(outfile, mask=None, debug=False,
                        mhw_sys=None, save_spat_root=None, clobber=False,
                       vmax=None, days=False, use_km=True):
    """
    Spatial distribution of MHW systems

    Args:
        outfile:
        mhw_events:
        events (xarray.DataArray):
        cube:
        save_spat_file (str, optional):
            Save the spatial info to this file as a .nc 
        clobber (bool, optional): 
            If True, clobber the spat_file (if provided)
        days (bool, optional):
            If True, count days instead of systems

    Returns:
        tuple: mask, mhw_sys
    """
    # Load MHW Systems
    if mhw_sys is None:
        mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019.csv')
        mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file)#, vary=vary)

    mask_file=os.path.join(os.getenv('MHW'), 'db', 'MHWS_2019_mask.nc')

    # km?
    if use_km:
        vox_key = 'NVox_km'
        area_key = 'max_area_km'
        type_dict = defs.type_dict_km
    else:
        vox_key = 'NVox'
        area_key = 'max_area'
        type_dict = defs.type_dict

    fig = plt.figure(figsize=(8.5, 11))
    plt.clf()
    proj = ccrs.PlateCarree(central_longitude=-180.0)
    gs = gridspec.GridSpec(3,1)

    for ss, mhw_type, cmap, vmax in zip(range(3),
        ['minor', 'normal', 'extreme'],
        ['Blues', 'Greens', 'Reds'],
        [200., 800., None]): 

        # Init
        NVox_mxn = type_dict[mhw_type]
        if save_spat_root is not None:
            save_spat_file = save_spat_root+mhw_type+'.nc'

        # Cut Systems
        cut = (mhw_sys[vox_key] >= NVox_mxn[0]) & (
            mhw_sys[vox_key] <= NVox_mxn[1])
        cut_sys = mhw_sys[cut]

        if days:
            sys_flag = np.zeros(mhw_sys.mask_Id.max()+1, dtype=int)
            sys_flag[cut_sys.mask_Id] = 1

        if save_spat_file is not None and os.path.isfile(
            save_spat_file) and not clobber:
            print(f"Loading data from {save_spat_file}")
            ds = xarray.open_dataset(save_spat_file) 
            spat_systems = ds.spatial
        else:
            # Mask
            if mask is None:
                if debug:
                    print("Loading only a subset of the mask for debuggin")
                    mask = mhw_sys_io.maskcube_from_slice(0, 4000)#, vary=True)
                else:
                    mask = mhw_sys_io.load_full_mask(mhw_mask_file=mask_file)#vary=True)

            # Lazy
            _ = mask.data
            print("C time")
            print(datetime.datetime.now())
            if not days:
                spat_systems = mhw_analysisc.spatial_systems(mask.data,
                                                        cut_sys.mask_Id.values.astype(np.int32),
                                                        np.max(mhw_sys.mask_Id.values))
            else:
                spat_systems = mhw_analysisc.days_in_systems(
                    mask.data, sys_flag.astype(np.int32))
            print(datetime.datetime.now())

            # Recast as xarray
            spat_systems = xarray.DataArray(spat_systems, coords=[mask.lat, mask.lon])

            # Save?
            if save_spat_file is not None and (
                not os.path.isfile(save_spat_file) or clobber):
                ds = xarray.Dataset({'spatial': spat_systems})
                ds.to_netcdf(save_spat_file, engine='h5netcdf')


        ax= plt.subplot(gs[ss], projection=proj)

        # Pacific events
        # Draw the contour with 25 levels.
        if days:
            cbar_lbl = 'Number'# of Days'
        else:
            cbar_lbl = 'Number of MHWS'
        #cm = plt.get_cmap('YlOrRd')
        cm = plt.get_cmap(cmap)
        p = spat_systems.plot(cmap=cm, transform=ccrs.PlateCarree(),
                            vmax=vmax, 
                            subplot_kws={'projection': proj},
                            cbar_kwargs={'label': cbar_lbl,
                                        'fraction': 0.020, 'pad': 0.04})
        ax = p.axes

        #cplt = iris.plot.contourf(cube_slice, 10, cmap=cm)  # , vmin=0, vmax=20)#, 5)
        #cb = plt.colorbar(cplt, fraction=0.020, pad=0.04)
        #cb.set_label('Average Annual Number of MHW Events')
        #cb.set_label('Number of MHW Events with t>1 month')

        # Gridlines
        # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5,
                        linestyle='--', draw_labels=True)
        gl.top_labels = False
        gl.left_labels = True
        gl.right_labels = False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
        gl.xlocator = mticker.FixedLocator([-240., -180., -120, -60, 0, 60, 120.])
        #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])

        # Add coastlines to the map created by contourf.
        ax.coastlines()

        # Turn off Title
        plt.title('')

    # Layout and save
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    return mask, mhw_sys

from numba import njit, prange
@njit(parallel=True)
def count_mhwe(ilon, jlat, duration, nevents, ndays):
    for kk in prange(ilon.size):
        nevents[jlat[kk],ilon[kk]] += 1
        ndays[jlat[kk],ilon[kk]] += duration[kk]

def fig_spatial_mhwe(outfile, mask=None, debug=False,
                        mhw_sys=None, save_spat_root=None, clobber=False,
                       vmax=None, days=False):
    """
    Spatial distribution of MHWEs

    Args:
        outfile:
        mhw_events:
        events (xarray.DataArray):
        cube:
        save_spat_file (str, optional):
            Save the spatial info to this file as a .nc 
        clobber (bool, optional): 
            If True, clobber the spat_file (if provided)
        days (bool, optional):
            If True, count days instead of systems

    Returns:
        tuple: mask, mhw_sys
    """
    # Load MHWE
    MHWE_path = os.path.join(os.getenv('MHW'), 'db')
    MHWE_file = os.path.join(MHWE_path, 'mhw_events_allsky_defaults.parquet')
    print("Loading..")
    mhwe = pandas.read_parquet(MHWE_file)

    # Load climate for coords
    noaa_path = os.getenv("NOAA_OI")
    climate_file = os.path.join(noaa_path, 'NOAA_OI_climate_1983-2019.nc') 
    clim = xarray.open_dataset(climate_file)

    # Convert lat/lon to indices
    angular_res=0.25 
    lon_lat_min=(0.125, -89.975)
    ilon = ((mhwe['lon'].values - lon_lat_min[0]) / angular_res).astype(np.int32)
    jlat = ((mhwe['lat'].values - lon_lat_min[1]) / angular_res).astype(np.int32)

    # Build straight days
    print("Counting..")
    nevents = np.zeros((clim.lat.size, clim.lon.size), dtype=int)
    ndays = np.zeros((clim.lat.size, clim.lon.size), dtype=int)
    count_mhwe(ilon, jlat, mhwe.duration.values, nevents, ndays)

    da_events = xarray.DataArray(nevents, coords=[clim.lat, clim.lon])
    da_days = xarray.DataArray(ndays, coords=[clim.lat, clim.lon])

    fig = plt.figure(figsize=(6, 9))
    plt.clf()
    proj = ccrs.PlateCarree(central_longitude=-180.0)
    gs = gridspec.GridSpec(2,1)

    for ss in range(2):
        ax= plt.subplot(gs[ss], projection=proj)

        # Draw the contour with 25 levels.
        if ss == 1:
            cbar_lbl = 'Number of Days'
            da = da_days
        else:
            cbar_lbl = 'Number of MHWS'
            da = da_events
        cm = plt.get_cmap('YlOrRd')
        p = da.plot(cmap=cm, transform=ccrs.PlateCarree(),
                            vmax=vmax, 
                            subplot_kws={'projection': proj},
                            cbar_kwargs={'label': cbar_lbl,
                                        'fraction': 0.020, 'pad': 0.04})
        ax = p.axes

        #cplt = iris.plot.contourf(cube_slice, 10, cmap=cm)  # , vmin=0, vmax=20)#, 5)
        #cb = plt.colorbar(cplt, fraction=0.020, pad=0.04)
        #cb.set_label('Average Annual Number of MHW Events')
        #cb.set_label('Number of MHW Events with t>1 month')

        # Gridlines
        # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5,
                        linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
        gl.xlocator = mticker.FixedLocator([-240., -180., -120, -60, 0, 60, 120.])
        #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])

        # Add coastlines to the map created by contourf.
        ax.coastlines()

        # Turn off Title
        plt.title('')

    # Layout and save
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    return mask, mhw_sys

def fig_detrend_global(outfile='fig_detrend_global.png'):
    # Load up
    detrend_file = os.path.join(noaa_path,
                                'NOAA_OI_detrend_local_climate_1983-2019.nc')
    ds = xarray.open_dataset(detrend_file)                        
    slope = ds.linear[0,:,:]
    slope.data *= 365 # Scale to per year

    # Figure
    fig = plt.figure(figsize=(6.5, 4))
    plt.clf()
    proj = ccrs.PlateCarree(central_longitude=-180.0)
    gs = gridspec.GridSpec(1,1)

    ax= plt.subplot(gs[0], projection=proj)

    cm = plt.get_cmap('seismic')
    p = slope.plot(cmap=cm, transform=ccrs.PlateCarree(),
                            #vmax=vmax, 
                            subplot_kws={'projection': proj},
                            add_colorbar=False)
                            #cbar_kwargs={'label': 'slope (deg K per year)',
                            #            'fraction': 0.020, 'pad': 0.04,
                            #            'fontsize': 15})
    cb = plt.colorbar(p, pad=0.04, fraction=0.020)
    cb.set_label(label='slope (deg K per year)',
                 fontsize=11) #weight='bold')
    cb.ax.tick_params(labelsize='small')

    ax = p.axes

    # Add coastlines to the map created by contourf.
    ax.coastlines()

    # Gridlines
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
                      alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    #gl.ylabels_left = True
    #gl.left_labels = True
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
    #gl.xlocator = mticker.FixedLocator([-240., -180., -120, -60, 0, 60, 120.])
    #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])
    gl.ylines = True


    # Turn off Title
    plt.title('')

    # Layout and save
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_compare_MHWE(outfile='fig_compare_MHWE.png', show_all=False):
    MHW_path = os.getenv('MHW')
    if show_all:
        MHWE_files = ['mhw_events_allsky_defaults.parquet', 
                    'mhw_events_allsky_2019.parquet',
                    'mhw_events_allsky_2019_median.parquet',
                    'mhw_events_allsky_2019_mean.parquet',
                    'mhw_events_allsky_2019_nosmooth.parquet',
                    'mhw_events_allsky_2019_mean_DT0.5.parquet', 
                    'mhw_events_allsky_2019_local.parquet',
                    'mhw_events_allsky_2019_DT0.5.parquet']
        lbls = ['Hobday', '2019', '2019 median de-trend',
                '2019 mean de-trend', '2019 No smooth', 
                '2019 mean de-trend; DT>0.5',
                '2019 local', '2019 with DT >0.5']
    else:
        MHWE_files = ['mhw_events_allsky_defaults.parquet', 
                  'mhw_events_allsky_2019.parquet',
                  'mhw_events_allsky_2019_local.parquet',
                  'mhw_events_allsky_2019_median.parquet',
                  ]
        lbls = ['Hobday', '2019', 
                '2019 local de-trend',
                '2019 global de-trend']
        clrs = ['k', 'b', 'r', 'orange']

    def count_em(mhwe, years):
        Nvox_year = np.zeros_like(years)
        N_MHWE_year = np.zeros_like(years)
        for ii, year in enumerate(years):
            tstart = datetime.datetime(year,1,1).toordinal()
            tend = datetime.datetime(year,12,31).toordinal()
            #
            in_year = (mhwe.time_peak >= tstart) & (mhwe.time_peak <= tend)
            # Sum
            Nvox = np.sum(mhwe[in_year].duration)
            Nvox_year[ii] = Nvox
            N_MHWE_year[ii] = np.sum(in_year)
        # Return
        return N_MHWE_year, Nvox_year
    
    years = np.arange(1983, 2020)
    
    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()

    for lbl, MHWE_file, clr in zip(lbls, MHWE_files, clrs):
        # Load
        mhwe_file = os.path.join(MHW_path, 'db', MHWE_file)
        print("Loading: {}".format(mhwe_file))
        mhwe = pandas.read_parquet(mhwe_file)
        N_MHWE, Nvox = count_em(mhwe, years)

        #ax.scatter(years, Nvox, label=lbl)
        #ax.plot(years, Nvox, label=lbl, color=clr)
        ax.plot(years, N_MHWE, label=lbl, color=clr)
    #
    ax.legend(loc='upper left', fontsize=15.)
    ax.set_yscale('log')
    #ax.set_ylim(4e6, 1e8)
    #
    ax.set_xlabel('Year')
    #ax.set_ylabel('N per year (vox)')
    ax.set_ylabel('Number of MHWE per year')

    set_fontsize(ax, 15.)

    # Layout and save
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def mk_segments(mapimg,dx,dy,x0=-0.5,y0=-0.5):
    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
    ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    try:
        segments[:, 0] = x0 + dx * segments[:, 0] / mapimg.shape[1]
        segments[:, 1] = y0 + dy * segments[:, 1] / mapimg.shape[0]
    except:
        embed(header='2346 of figs')

    return segments

def add_gridlines(ax):
    # Gridlines
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5,
                      linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}

    # gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
    # gl.xlocator = mticker.FixedLocator([-240., -180., -120, -60, 0, 60, 120.])
    # gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])


def set_mplrc():
    mpl.rcParams['mathtext.default'] = 'it'
    mpl.rcParams['font.size'] = 12
    mpl.rc('font',family='Times New Roman')
    mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
    mpl.rc('text', usetex=True)


def set_fontsize(ax, fsz):
    """
    Set the fontsize throughout an Axis

    Args:
        ax (Matplotlib Axis):
        fsz (float): Font size

    Returns:

    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)

def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x, y : array_like (nsamples,)
       The samples.

    bins : int or list

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes (optional)
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    plot_density : bool (optional)
        Draw the density colormap.

    plot_contours : bool (optional)
        Draw the contours.

    no_fill_contours : bool (optional)
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool (optional)
        Fill the contours.

    contour_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.
    """
    from matplotlib.colors import LinearSegmentedColormap, colorConverter
    from scipy.ndimage import gaussian_filter

    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=range, weights=weights)
    except ValueError:
        embed(header='732 of figs')
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        print("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])


#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    if flg_fig & (2 ** 0):
        if False:
            '''
            import pandas
            import sqlalchemy
            mhw_file = '/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db'
            engine = sqlalchemy.create_engine('sqlite:///' + mhw_file)
            # Load
            mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                               columns=['date', 'lon', 'lat', 'duration',
                                                        'ievent', 'time_start', 'index', 'category'])
            '''
            print("Loading events..")
            mhw_file = '/home/xavier/Projects/Oceanography/MHW/db/mhw_events_allsky_vary.hdf'
            mhw_events = pandas.read_hdf(mhw_file, 'MHW_Events')
            print("Loaded!")
        else:
            mhw_events = None

        # Load events (1 month duration)
        events = xarray.load_dataarray('events_1month.nc')
        for outfile in ['fig_mhw_events.png']: #, 'fig_mhw_events.pdf']:
                fig_mhw_events(outfile,
                               events=events,
                               mhw_events=mhw_events,
                               events_file=None, #'events_1month.nc',
                               duration=30)
            #fig_mhw_events(outfile, cube=cube)

    if flg_fig & (2 ** 1):
        for outfile in ['fig_pacific_blob.png', 'fig_pacific_blob.pdf']:
            fig_pacific_blob(outfile)

    if flg_fig & (2 ** 2):
        for outfile in ['fig_NVox_by_year.png', 'fig_NVox_by_year.pdf']:
            fig_NVox_by_year(outfile, lbl='All Latitudes', vary=True)

        # Ignore high latitude
        #mhw_sys_file = os.path.join(os.getenv('MHW'), 'db', 'MHW_systems_nohilat.hdf')
        #for outfile in ['fig_NSpax_by_year_nohi.png']:
        #    fig_NSpax_by_year(outfile, lbl='Ignore High Latitudes', mhw_sys_file=mhw_sys_file)

    # NSpax histograms
    if flg_fig & (2 ** 3):
        for outfile in ['fig_NVox_hist.png']:#, 'fig_NVox_hist.pdf']:
            fig_NVox_hist(outfile)

    # duration histograms
    if flg_fig & (2 ** 4):
        for outfile in ['fig_duration_hist.png']:#, 'fig_duration_hist.pdf']:
            fig_duration_hist(outfile)

    # max area histograms
    if flg_fig & (2 ** 5):
        for outfile in ['fig_maxarea_hist.png']:#, 'fig_maxarea_hist.pdf']:
            fig_maxarea_hist(outfile)

    # Duration vs. NVox
    if flg_fig & (2 ** 6):
        for outfile in ['fig_durvsNVox.png']:#, 'fig_durvsNVox.pdf']:
            fig_dur_vs_NVox(outfile)

    # MHW Events vs. year
    if flg_fig & (2 ** 7):
        for outfile in ['fig_mhw_events_time.png']: #, 'fig_mhw_events_time.pdf']:
            fig_mhw_events_time(outfile, load_full=True)

    # Climate
    if flg_fig & (2 ** 8):
        for outfile in ['fig_climate.png']: #, 'fig_mhw_events_time.pdf']:
            fig_climate(outfile)

    # MaxArea vs. NVox
    if flg_fig & (2 ** 9):
        for outfile in ['fig_maxA_vs_NVox.png', 'fig_maxA_vs_NVox.pdf']:
            fig_maxA_vs_NVox(outfile)

    # Location
    if flg_fig & (2 ** 10):
        for ext, size, nside, nmax in zip(['png', 'png'],
                                    ['low', 'int'],
                                    [128, 32],
                                    [20, 5],
                                    ):
            fig_location_NVox(ext, size, nside=nside, nmax=nmax)

    # Example MHWS
    if flg_fig & (2 ** 11):
        for outfile in ['fig_example_mhws.png']:
            fig_example_mhws(outfile, make_mayavi=False)

    # Nsys vs. year
    if flg_fig & (2 ** 12):
        for outfile in ['fig_Nsys_by_year.png']:
            fig_NSys_by_year(outfile)

    # Nsys vs. year
    if flg_fig & (2 ** 13):
        #fig_spatial_sytems('fig_spatial_intermediate.png', (1e3, 1e5), debug=True,
        #                   save_spat_file='spatial_intermediate.nc')
        for mhw_type, vmax in zip(['random', 'normal', 'extreme'],
                                  [40., None, None]): 
            fig_spatial_systems('fig_spatial_{}.png'.format(mhw_type), 
                           defs.type_dict[mhw_type],
                           debug=False, vmax=vmax,
                           save_spat_file='spatial_{}.nc'.format(mhw_type))
        #fig_spatial_sytems('fig_spatial_extreme.png', 
        #                   (1e5, 1e9), 
        #                   debug=False,
        #                   save_spat_file='spatial_extreme.nc')

    # Tthresh vs. DOY
    if flg_fig & (2 ** 14):
        fig_T_thresh('fig_T_thresh.png')

    # Nvox vs. year by type and method
    if flg_fig & (2 ** 15):
        for use, outfile in zip([False,True], ['fig_Nvox_by_year.png',
                                               'fig_Nvox_by_year_km.png']):
            if not use:
                continue                                   
            fig_Nvox_by_year(outfile, use_km=use)

    # Gallery of intermediate events
    if flg_fig & (2 ** 16):
        '''
        # Vary MHW
        rand_dict = dict(min_dur=datetime.timedelta(days=60), type='intermediate', seed=12149)
        fig_int_gallery('MHW_systems_vary.csv', 'MHW_mask_vary.nc', 'fig_int_gallery_vary.png', rand_dict)
        '''

        # Cold spells
        rand_dict = dict(min_dur=datetime.timedelta(days=60), type='intermediate', seed=12149)
        fig_int_gallery('MCS_systems.csv', 'MCS_mask.nc', 'fig_int_gallery_MCS.png', rand_dict)

    # Extreme gallery
    if flg_fig & (2 ** 17):
        #fig_extreme_gallery('MHW_systems_vary.csv', 'MHW_mask_vary.nc', 'fig_extreme_gallery', next=10)
        #fig_extreme_gallery('MHW_systems_vary_95.csv', 'MHW_mask_vary_95.nc', 'fig_extreme_gallery_95', next=10)
        fig_extreme_gallery('MCS_systems.csv', 'MCS_mask.nc', 'fig_extreme_gallery_MCS', next=10, cold=True)

    # SST vs. Tthresh
    if flg_fig & (2 ** 18):
        fig_SST_vs_Tthresh('fig_SST_vs_Tthresh')

    # MHWS Histograms
    if flg_fig & (2 ** 19):
        #fig_MHWS_histograms('fig_MHWS_histograms.png', use_km=False)
        fig_MHWS_histograms('fig_MHWS_histograms_km.png')

    # Days in a given system vs. location
    if flg_fig & (2 ** 20):
        fig_spatial_systems(
            'fig_days_systems_km.png',
            debug=False, days=True, clobber=False, 
            save_spat_root='days_spatial_km_')
        #fig_spatial_systems(
        #    'fig_days_systems.png',
        #    debug=False, days=True, clobber=False, 
        #    save_spat_root='days_spatial_', use_km=False)
        #fig_spatial_sytems('fig_spatial_extreme.png', 
        #                   (1e5, 1e9), 
        #                   debug=False,
        #                   save_spat_file='spatial_extreme.nc')

    # Extreme evolution
    if flg_fig & (2 ** 21):
        fig_extreme_evolution('MHW_systems_vary.csv', 'MHW_mask_vary.nc', 
            0, 'fig_extreme_ex0.png', lon_mnx=(120., 300.),
            lat_mnx=(-70., 70.))
        #fig_extreme_gallery('MHW_systems_vary_95.csv', 'MHW_mask_vary_95.nc', 'fig_extreme_gallery_95', next=10)
        #fig_extreme_gallery('MCS_systems.csv', 'MCS_mask.nc', 'fig_extreme_gallery_MCS', next=10, cold=True)

    # Compare MHWE
    if flg_fig & (2 ** 22):
        fig_compare_MHWE()

    # Days in a given MHWE vs. location
    if flg_fig & (2 ** 23):
        fig_spatial_mhwe(
            'fig_days_mhwe.png',
            debug=False, days=True, clobber=False) 

    # Compare MHWE
    if flg_fig & (2 ** 24):
        fig_detrend_global()

    # Cumulative NVox
    if flg_fig & (2 ** 25):
        fig_cumulative_NVox('fig_cumulative_NVox_km.png', use_km=True)


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # MHW Events (spatial)
        #flg_fig += 2 ** 1  # Pacific blob
        #flg_fig += 2 ** 2  # Total, median (NVox) vs. year
        #flg_fig += 2 ** 3  # NVox histograms
        #flg_fig += 2 ** 4  # Duration histograms
        #flg_fig += 2 ** 5  # max area histograms
        #flg_fig += 2 ** 6  # duration vs. NVox
        #flg_fig += 2 ** 7  # MHW Events (time) -- THIS CAN BE VERY SLOW
        #flg_fig += 2 ** 8  # Climate
        #flg_fig += 2 ** 9  # max area vs. NSpax
        #flg_fig += 2 ** 10  # Location location location
        flg_fig += 2 ** 11  # Example MHWS
        #flg_fig += 2 ** 12  # Nsys vs. year
        #flg_fig += 2 ** 13  # Spatial location of Systems
        #flg_fig += 2 ** 14  # Tthresh, T90, T95 vs DOY
        #flg_fig += 2 ** 15  # Nvox vs. year by method
        #flg_fig += 2 ** 16  # Intermediate gallery
        #flg_fig += 2 ** 17  # Extreme examples
        #flg_fig += 2 ** 18  # SST vs. T_thresh
        #flg_fig += 2 ** 19  # Main Histogram figure
        #flg_fig += 2 ** 20  # Spatial in days
        #flg_fig += 2 ** 21  # Extreme evolution
        #flg_fig += 2 ** 22  # Comparing MHWE definitions/approaches (by year)
        #flg_fig += 2 ** 23  # MHWE spatial
        #flg_fig += 2 ** 24  # de-trend global view
        #flg_fig += 2 ** 25  # Cumulative NVox
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
