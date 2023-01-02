""" Figures for the first paper on MHW Systems"""
import os, sys
import numpy as np
from pkg_resources import resource_filename

from datetime import date, datetime

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, use
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

mpl.rcParams['font.family'] = 'splttixgeneral'

import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import cartopy

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
import defs, analy_utils, fitting, analy_sys


def fig_t_cdf(outfile='fig_t_cdf.png', 
              mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
              vary=False):
    """
    """
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=vary)
    tdur_days = mhw_sys.duration.values / np.timedelta64(1, 'D')
    srt_mhws = np.argsort(tdur_days)

    mhws_cdf = (1+np.arange(tdur_days.size)) / tdur_days.size

    # Figure
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    ax = plt.gca()

    ax.plot(tdur_days[srt_mhws], mhws_cdf, 'k-', lw=2)

    # Axes
    ax.set_xlabel('t (days)')
    ax.set_ylabel('CDF')
    ax.set_xscale('log')
    ax.set_yscale('log')

    set_fontsize(ax, 21.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_t_histograms(outfile='fig_t_histograms.png', 
              mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
              vary=False):
    """
    """
    # Load MHWE
    MHW_path = os.getenv('MHW')
    MHWE_file = 'mhw_events_allsky_2019.parquet'
    mhwe_file = os.path.join(MHW_path, 'db', MHWE_file)
    print('Loading MHWE')
    mhwe = pandas.read_parquet(mhwe_file)
    print('Done')

    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=vary)
    # Convert to days
    tdur_days = mhw_sys.duration.values / np.timedelta64(1, 'D')
    mhw_sys['duration'] = tdur_days

    # Figure
    fig = plt.figure(figsize=(10, 8))
    plt.clf()
    ax = plt.gca()

    bins = np.linspace(np.log10(5), 4.0, 32)
    attr = 'duration'
    if np.log10(np.max(mhw_sys[attr].values)+1) > bins[-1]:
        bins[-1] = np.log10(np.max(mhw_sys[attr].values)+1) # Extend to incluce the last one

    # MHWS
    sns.histplot(mhw_sys, x=attr, bins=bins, log_scale=True, ax=ax,
                     color='g', label='MHWS')
    # MHWE
    sns.histplot(mhwe, x='duration', bins=bins, log_scale=True, ax=ax,
                 color='blue', label='MHWE')

    # Axes
    ax.set_xlabel(r'$t_{\rm dur}$ (days)')
    ax.set_ylabel('Number')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.legend(fontsize=21.)

    set_fontsize(ax, 21.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_average_area(outfile='fig_average_area.png', 
              mhw_sys_file=os.path.join(os.getenv('MHW'), 'db', 
                            'MHWS_2019.csv'),
              vary=False):
    # Load MHW Systems
    mhw_sys = mhw_sys_io.load_systems(mhw_sys_file=mhw_sys_file, 
                                      vary=vary)
    # Convert to days
    tdur_days = mhw_sys.duration.values / np.timedelta64(1, 'D')
    mhw_sys['duration'] = tdur_days
    mhw_sys['avg_area'] = mhw_sys.NVox_km.values / tdur_days
    #embed(header='153 of discuss')

    # Remove the 4's
    good = mhw_sys.avg_area.values > 4.
    mhw_sys = mhw_sys[good].copy()

    # Figure
    fig = plt.figure(figsize=(10, 8))
    plt.clf()
    ax = plt.gca()

    nbins = 50
    bins_dur = np.linspace(0.5, 4.0, nbins)
    bins_aarea = np.linspace(0.5, 7.5, nbins)

    counts, xedges, yedges = np.histogram2d(np.log10(mhw_sys.duration.values),
        np.log10(mhw_sys.avg_area.values), bins=(bins_dur, bins_aarea))
    cm = plt.get_cmap('autumn')
    mplt = ax.pcolormesh(10**xedges,10**yedges,
                                np.log10(counts.transpose()), cmap=cm)
    cb = plt.colorbar(mplt, fraction=0.030, pad=0.04)
    cb.ax.tick_params(labelsize=13.)
    cb.set_label(r'log10 Counts', fontsize=20.)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t_{\rm dur}$ (days)')
    ax.set_ylabel(r'$\bar A$ (km$^2$)')

    '''
    jg = sns.jointplot(data=mhw_sys, x='duration', y='avg_area', kind='hex',
                       bins='log', gridsize=100, xscale='log', yscale='log',
                       cmap=plt.get_cmap('autumn'), mincnt=1)
                       #marginal_kws=dict(fill=False, color='black', 
                       #                  bins=100)) 
    plt.colorbar()
    jg.ax_marg_x.set_axis_off()
    jg.ax_marg_y.set_axis_off()
    # Axes                                 
    #jg.ax_joint.set_xlabel(r'$\Delta T$')
    #jg.ax_joint.set_ylim(ymnx)
    #jg.fig.set_figwidth(8.)
    #jg.fig.set_figheight(7.)

    set_fontsize(jg.ax_joint, 16.)
    '''

    set_fontsize(ax,  16.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    # Stats
    # t=50-100
    t50 = mhw_sys[(mhw_sys.duration > 50) & (mhw_sys.duration < 100)]
    p10_t50 = np.percentile(t50.avg_area.values, 10)
    mean_t50 = np.mean(t50.avg_area.values)
    print(f"10th percentile of average area for t=50-100: {p10_t50}")
    print(f"Mean of average area for t=50-100: {mean_t50}")

    t1year = mhw_sys[(mhw_sys.duration > 365)]
    p10_t1year = np.percentile(t1year.avg_area.values, 10)
    print(f"10th percentile of average area for t>1year: {p10_t1year}")


def set_mplrc():
    mpl.rcParams['mathtext.default'] = 'it'
    mpl.rcParams['font.size'] = 12
    mpl.rc('font',family='Times New Roman')
    #mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
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


#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # t CDF (not so interesting)
    if flg_fig & (2 ** 0):
        fig_t_cdf()

    # t histograms
    if flg_fig & (2 ** 1):
        fig_t_histograms()

    # Average area
    if flg_fig & (2 ** 2):
        fig_average_area()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # CDF of durations
        #flg_fig += 2 ** 1  # t histograms
        #flg_fig += 2 ** 2  # Average area
        flg_fig += 2 ** 3  # <A> vs year
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)