""" Driver for the build which uses either Python or C"""
import numpy as np
import warnings

import sqlalchemy
import pandas as pd

from mhw_analysis.systems import buildpy
from mhw_analysis.systems import buildc

from IPython import embed

try:
    from mhw_analysis.systems.buildc import define_systems
except:
    warnings.warn('Unable to load build C extension.  Try rebuilding mhw_analysis.  In the '
                  'meantime, falling back to pure python code.')
    from mhw_analysis.systems.buildpy import define_systems

def first_pass():

    # Load
    cube = np.load('../../doc/nb/tst_cube_pacific.npy')

    # C
    maskC, parentC = buildc.first_pass(cube.astype(bool))
    #embed(header='25 of build')

    print("Done with C")

    # Python
    mask, parent = buildpy.define_systems(cube.astype(bool), return_first=True)

    embed(header='32 of build')


def full_test():
    """
    Python
    """
    #cube = np.load('../../doc/nb/tst_cube.npy')
    cube = np.load('../../doc/nb/tst_cube_pacific.npy')
    mask, obj_dict = buildpy.define_systems(cube.astype(bool))
    # Write
    #np.save('tst_mask', mask)
    np.save('pacific_mask', mask)
    # Pandas
    df = pd.DataFrame(obj_dict)
    # SQL
    #dbfile = '/home/xavier/Projects/Oceanography/MHW/db/tst_mhw_systems.db'
    dbfile = '/home/xavier/Projects/Oceanography/MHW/db/pacific_mhw_system.db'
    engine = sqlalchemy.create_engine('sqlite:///'+dbfile)
    df.to_sql('MHW_Systems', con=engine)#, if_exists='append')

# Testing
if __name__ == '__main__':
    #full_test()

    # C testing
    first_pass()

