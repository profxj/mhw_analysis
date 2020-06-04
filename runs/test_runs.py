from mhw_analysis.db import build_mhws

def dont_run_this():
    # Test from Ipython
    from importlib import reload
    import glob
    from mhw_analysis.db import build_mhws
    from mhw_analysis.db import utils
    noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/'
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()
    #years = [1982,2016]
    #years = [1986,1990]
    years = [1983,2012]
    istart = years[0]-1981
    iend = years[1]-1981+1
    all_sst_files = all_sst_files[istart:iend]
    all_sst = utils.load_noaa_sst(all_sst_files)
    #
    reload(build_mhws)
    build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/db/mhws_allsky_defaults.db', years=years,
                        cut_sky=True, all_sst=all_sst, append=False)#, n_calc=1000)
    #build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', years=years,
    #                    cut_sky=False, all_sst=all_sst, nproc=50, n_calc=1000)

# Command line execution
if __name__ == '__main__':
    #
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.db', cut_sky=True)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws_allsky.db', cut_years=True, cut_sky=False)
    #build_me('/home/xavier/Projects/Oceanography/MHWs/db/test_mhws_allsky.db', years=[1982,2016], cut_sky=False, nproc=50, n_calc=1000)

    # Default run to match Oliver (+ a few extra years)
    build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/mhws_allsky_defaults.db',
                        years=[1982,2019], cut_sky=False, nproc=50)

