""" Driver for the build which uses either Python or C"""
import numpy as np
import warnings

import sqlalchemy
import pandas as pd
import datetime
import h5py

import iris

from mhw_analysis.systems import buildpy
from mhw_analysis.systems import buildc
from mhw_analysis.systems import utils

from IPython import embed

try:
    from mhw_analysis.systems.buildc import final_pass
except:
    warnings.warn('Unable to load build C extension.  Try rebuilding mhw_analysis.  In the '
                  'meantime, falling back to pure python code.')
    from mhw_analysis.systems.buildpy import final_pass

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


    # C
    buildc.max_areas(maskC, obj_dictC)

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


def main(sub=None, mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems.hdf',
         cube=None, ymd_start=(1982, 1, 1), ignore_hilat=False):
    """
    Generate MHW Systems from an Event cube

    Args:
        sub (tuple, optional):
            Restrict run to a subset of dates
        mhwsys_file (str, optional):
            Output file
        cube (numpy.ndarray):
        ymd_start:
        ignore_hilat:

    Returns:

    """
    # Load cube -- See MHW_Cube Notebook
    if cube is None:
        print("Loading cube")
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube.nc'
        cubes = iris.load(cubefile)
        cube = cubes[0].data[:].astype(np.int8)
        print("Cube is loaded")

    # Ignore high latitude events
    if ignore_hilat:
        print("Ignoring high latitude events (|b| > 65 deg)")
        cube[0:100,:,:] = 0
        cube[-100:,:,:] = 0

    # Sub?
    if sub is not None:
        cube = cube[:,:,sub[0]:sub[1]].astype(np.int8)
        print("Sub")

    # C
    maskC, parentC, catC = buildc.first_pass(cube)
    # maskC[558,857,:]
    print("First pass complete")
    NSpaxC = buildc.second_pass(maskC, parentC, catC)
    print("Second pass complete")
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NSpaxC, MinNSpax=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NSpaxC, ndet, IdToLabel, LabelToId, catC)
    print("Objects nearly done")

    # Area
    buildc.max_areas(maskC, obj_dictC)
    print("area done")

    # Write systems as pandas in HDF
    tbl = utils.dict_to_pandas(obj_dictC, add_latlon=True,
                               start_date=datetime.date(ymd_start[0], ymd_start[1], ymd_start[2]).toordinal())
    tbl.to_hdf(mhwsys_file, 'mhw_sys', mode='w')
    print("Wrote: {}".format(mhwsys_file))

    # Write mask as HDF
    mask_file = mhwsys_file.replace('systems', 'mask')
    f = h5py.File(mask_file, mode='w')
    dset = f.create_dataset("mask", #maskC.shape, dtype='int32',
                            compression='gzip', data=maskC,
                            chunks=(maskC.shape[0], maskC.shape[1], 1))
    f.close()
    #mask_file = mask_file.replace('hdf', 'npz')
    #np.savez_compressed(mask_file, mask=maskC)
    print("Wrote: {}".format(mask_file))


    # Return
    return tbl, maskC

# Testing
if __name__ == '__main__':
    #full_test()

    # C testing
    #test_c()

    # Debuggin
    #tbl, mask = main(sub=(11600,11600+380), mhwsys_file='tst.hdf', ymd_start=(2013, 10, 5))
    if False:
        cube = np.load('tst_cube.npz')['arr_0'].astype(np.int8)
        # Zero out high/low latitudes
        cube[0:100,:,:] = 0
        cube[-100:,:,:] = 0
        tbl, mask = main(cube=cube, mhwsys_file='tst_systems.hdf', ymd_start=(2013, 10, 5))
        embed(header='134 of build')

    # Testing
    #main(sub=(10000,11000))

    # Real deal
    #main(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_nohilat.hdf',
    #     ignore_hilat=True)
    if False:
        cube = np.zeros((720,1440,100), dtype=np.int8)
        # Indian/Pacific
        cube[100:400,350:650,30:50] = 1
        # Indian/Atlantic
        cube[150:300,70:90,20:30] = 1
        # Pacific/Atlantic
        cube[150:300,1150:1170,10:20] = 1
        # Pacific/Atlantic
        cube[600:640,600:650,15:25] = 1
        # Run
        main(cube=cube, mhwsys_file='tst_indian_systems.hdf')

    if False:
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc'
        print("Loading: {}".format(cubefile))
        cubes = iris.load(cubefile) #np.load(cubefile)['cube'].astype(np.int8)
        cube = cubes[0].data[:,:,9500:12000].astype(np.int8)
        #
        main(mhwsys_file='test_basins_systems.hdf', cube=cube)

    # Testing
    #main(sub=(10000,11000))

    # Original
    if True:
        main(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_nohilat.hdf',
             ignore_hilat=True)
        main()

    # Vary
    if True:
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc'
        print("Loading: {}".format(cubefile))
        cubes = iris.load(cubefile) #np.load(cubefile)['cube'].astype(np.int8)
        cube = cubes[0].data[:].astype(np.int8)
        #
        main(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_vary.hdf', cube=cube)
