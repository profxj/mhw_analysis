from mhw_analysis import build_mhws


# Command line execution
if __name__ == '__main__':

    # Default run to match Oliver (+ a few extra years)
    build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/db/mhws_allsky_defaults.db',
                        years=[1982,2019], cut_sky=False, nproc=50)

