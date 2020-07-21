""" Driver for the build which uses either Python or C"""
import numpy as np
import warnings

import sqlalchemy
import pandas as pd

from mhw_analysis.systems import buildpy

try:
    from mhw_analysis.systems.buildc import define_systems
except:
    warnings.warn('Unable to load build C extension.  Try rebuilding mhw_analysis.  In the '
                  'meantime, falling back to pure python code.')
    from mhw_analysis.systems.buildpy import define_systems

# Testing
if __name__ == '__main__':
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
