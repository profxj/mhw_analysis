from mhw_analysis.db import build_mhws
from mhw_analysis.db import build_climate


# Command line execution
if __name__ == '__main__':

    # Default run to match Oliver (+ a few extra years)
    if True:
        build_mhws.build_me('/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db',
                        years=[1982,2019], cut_sky=False, append=False)

    # Climate
    if False:
        build_climate.build_noaa('/home/xavier/Projects/Oceanography/MHWs/db/NOAA_OI_climate_1983-2012.nc',
                                 cut_sky=False)

