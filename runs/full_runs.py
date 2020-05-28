from mhw_analysis.db import mhw
from mhw_analysis.db import build_climate


# Command line execution
if __name__ == '__main__':

    # Default run to match Oliver (+ a few extra years)
    if False:
        mhw.build_me('/home/xavier/Projects/Oceanography/MHWs/db/mhws_allsky_defaults.db',
                        years=[1982,2019], cut_sky=False, nproc=50)

    # Climate
    if True:
        build_climate.build_noaa('/home/xavier/Projects/Oceanography/MHWs/db/NOAA_OI_climate_1983-2012.nc',
                                 cut_sky=False)

