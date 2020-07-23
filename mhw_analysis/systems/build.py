""" Driver for the build which uses either Python or C"""
import numpy as np
import warnings

import sqlalchemy
import pandas as pd

from mhw_analysis.systems import buildpy
from mhw_analysis.systems import buildc
from mhw_analysis.systems import utils

from IPython import embed

try:
    from mhw_analysis.systems.buildc import define_systems
except:
    warnings.warn('Unable to load build C extension.  Try rebuilding mhw_analysis.  In the '
                  'meantime, falling back to pure python code.')
    from mhw_analysis.systems.buildpy import define_systems

def test_c():

    # Load
    cube = np.load('../../doc/nb/tst_cube_pacific.npy')

    # C
    maskC, parentC = buildc.first_pass(cube.astype(bool))
    NSpaxC = buildc.second_pass(maskC, parentC)
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NSpaxC, MinNSpax=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NSpaxC, ndet, IdToLabel, LabelToId)

    print("Done with C")

    # Python
    #mask, parent = buildpy.define_systems(cube.astype(bool), return_first=True)
    #mask, parent, NSpax = buildpy.define_systems(cube.astype(bool), return_second=True, verbose=True)
    mask, obj_dict = buildpy.define_systems(cube.astype(bool), verbose=True)

    assert np.array_equal(obj_dictC['zcen'], obj_dict['zcen'])


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
    test_c()

