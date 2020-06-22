import glob
import numpy as np
from importlib import reload

from mhw_analysis.db import utils
from mhw_analysis.db import build_climate
from mhw import climate

from IPython import embed

def load_me(years = (1983,1993)):
    # Grab the list of SST V2 files
    noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/'
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()
    # Cut on years
    if '1981' not in all_sst_files[0]:
        raise ValueError("Years not in sync!!")

    # Load the Cubes into memory
    istart = years[0] - 1981
    iend = years[1] - 1981 + 1
    all_sst_files = all_sst_files[istart:iend]

    print("Loading up the files. Be patient...")
    all_sst = utils.load_noaa_sst(all_sst_files)

    return all_sst


def first_try():
    #
    # Time
    all_sst = load_me()
    t = utils.grab_t(all_sst)

    time_dict = climate.build_time_dict(t)
    irange = np.arange(355, 365)
    jrange = np.arange(715, 725)
    ilat, jlon = irange[0], jrange[0]
    SST = utils.grab_T(all_sst, ilat, jlon)

    clim = climate.calc(time_dict, SST)

def test_jit():
    #sst = np.load('ex_SST.npy')
    #t = np.load('t.npy')

    sst = np.load('tempClim.np.npy')
    t = np.load('t2.np.npy')
    times = climate.build_time_dict(t)
    #

    all_temps = np.outer(np.ones(1000), sst)

    # Length of climatological year
    lenClimYear = 366
    feb29 = 60
    windowHalfWidth=5
    pctile = 90
    wHW_array = np.outer(np.ones(1000, dtype='int'), np.arange(-windowHalfWidth, windowHalfWidth + 1))
    # Inialize arrays
    thresh_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')
    seas_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')

    for kk in range(all_temps.shape[0]):
        print('kk: {}'.format(kk))
        tempClim = all_temps[kk,:]
        doyClim = times['doy']
        TClim = len(doyClim)

        # Start and end indices
        # clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
        # clim_end = np.where(yearClim == climatologyPeriod[1])[0][-1]
        clim_start = 0
        clim_end = len(doyClim)

        # clim['thresh'] = np.NaN*np.zeros(TClim)
        # clim['seas'] = np.NaN*np.zeros(TClim)
        nwHW = wHW_array.shape[1]

        climate.doit(lenClimYear, feb29, doyClim, clim_start, clim_end, wHW_array, nwHW,
         TClim, thresh_climYear, tempClim, pctile, seas_climYear)
        embed(header='81 of test')

# Command line execution
if __name__ == '__main__':
    #
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.db', cut_sky=True)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws_allsky.db', cut_years=True, cut_sky=False)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', years=[1982,2016], cut_sky=False, nproc=50, n_calc=1000)

    if False:
        test_jit()

    # Default run to match Oliver (+ a few extra years)
    if True:
        climatologyPeriod = [1983, 2012]
        all_sst = load_me(years=climatologyPeriod)
        embed(header='97 of test')
        build_climate.build_noaa('/home/xavier/Projects/Oceanography/MHWs/db/test_climate.nc',
                                 climatologyPeriod=climatologyPeriod,
                                 cut_sky=True, all_sst=all_sst)

