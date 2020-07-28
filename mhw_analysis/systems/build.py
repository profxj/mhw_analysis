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

    # Muck around
    good = cube == 1
    ngood = np.sum(good)
    cube[good] = np.random.randint(4, size=ngood)+1

    # C
    maskC, parentC, catC = buildc.first_pass(cube.astype(np.int8))
    NSpaxC = buildc.second_pass(maskC, parentC, catC)
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NSpaxC, MinNSpax=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NSpaxC, ndet, IdToLabel, LabelToId, catC)

    # Area
    obj_id = np.unique(maskC[maskC > 0])
    areas = np.zeros_like(obj_id)
    utils.max_area(maskC, obj_id, areas)

    obj_dictC['max_area'] = areas

    # pandas
    tbl = utils.dict_to_pandas(obj_dictC, add_latlon=True)
    tbl.to_hdf('tst.hdf', 'mhw_sys', mode='w')
    print("Wrote tst.hdf")

    print("Done with C")

    # Python
    #mask, parent = buildpy.define_systems(cube.astype(bool), return_first=True)
    #mask, parent, NSpax = buildpy.define_systems(cube.astype(bool), return_second=True, verbose=True)
    #mask, obj_dict = buildpy.define_systems(cube.astype(bool), verbose=True)

    #assert np.array_equal(obj_dictC['zcen'], obj_dict['zcen'])


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

def main(sub=None, mhwsys_file = '/home/xavier/Projects/Oceanography/MHW/db/MHW_systems.hdf'):
    # Load cube -- See MHW_Cube Notebook
    print("Loading cube")
    cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube.npz'
    cube = np.load(cubefile)['cube'].astype(np.int8)
    print("Cube is loaded")

    # Sub?
    if sub is not None:
        cube = cube[:,:,0:sub].astype(np.int8)
        print("Sub")

    # C
    maskC, parentC, catC = buildc.first_pass(cube)
    print("First pass complete")
    NSpaxC = buildc.second_pass(maskC, parentC, catC)
    print("Second pass complete")
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NSpaxC, MinNSpax=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NSpaxC, ndet, IdToLabel, LabelToId, catC)

    # Area
    utils.max_area(maskC, obj_dictC)

    # pandas
    tbl = utils.dict_to_pandas(obj_dictC, add_latlon=True)
    tbl.to_hdf(mhwsys_file, 'mhw_sys', mode='w')
    print("Wrote: {}".format(mhwsys_file))

    # Write
    #np.savez(mhwsys_file, **obj_dictC)
    mask_file = mhwsys_file.replace('systems', 'mask')
    np.savez_compressed(mask_file, mask=maskC)
    print("Wrote: {}".format(mask_file))

# Testing
if __name__ == '__main__':
    #full_test()

    # C testing
    test_c()

    # Real deal
    #main(sub=2500, mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_2000.npz')
    #main()

