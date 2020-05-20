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


# Command line execution
if __name__ == '__main__':
    #
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.db', cut_sky=True)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws_allsky.db', cut_years=True, cut_sky=False)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', years=[1982,2016], cut_sky=False, nproc=50, n_calc=1000)
    # Default run to match Oliver (+ a few extra years)
    climatologyPeriod = [1983, 1993]
    all_sst = load_me(years=climatologyPeriod)
    embed(header='55 of test')
    build_climate.build_noaa('/home/xavier/Projects/Oceanography/MHWs/db/test_climate.nc',
                             climatologyPeriod=climatologyPeriod,
                             cut_sky=True, all_sst=all_sst)

