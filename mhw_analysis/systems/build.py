""" Driver for the build which uses either Python or C"""
import os
import numpy as np
import warnings

from importlib import reload

import pandas as pd
import datetime

import xarray

from oceanpy.sst import utils as sst_utils

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

mhw_path = os.getenv('MHW')
mhwdb_path = os.path.join(mhw_path, 'db')

def test_c():
    """
    Simple method to test some of the C code
    """

    # Load
    cube = np.load('../../doc/nb/tst_cube_pacific.npy')

    # Muck around
    good = cube == 1
    ngood = np.sum(good)
    cube[good] = np.random.randint(4, size=ngood)+1

    # C    da = xarray.DataArray(maskC, coords=[lat_coord, lon_coord, times],
                          dims=['lat', 'lon', 'time'])
    ds = xarray.Dataset({'mask': da})
    print("Saving mask..")
    encoding = {'mask': dict(compression='gzip',
                         chunksizes=(maskC.shape[0], maskC.shape[1], 1))}
    ds.to_netcdf(mask_file, engine='h5netcdf', encoding=encoding)
    maskC, parentC, catC = buildc.first_pass(cube.astype(np.int8))
    NVoxC = buildc.second_pass(maskC, parentC, catC)
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NVoxC, MinNVox=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NVoxC, ndet, IdToLabel, LabelToId, catC)


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

'''
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
'''


def generate_mhw(mhwsys_file:str, sub=None, 
         cube=None, ymd_start=(1982, 1, 1), ignore_hilat=False, debug=False):
    """
    Generate MHW Systems from an Event data cube

    The list of MHWS is written to the input CSV file
    In addition, a cube describing the locations in lat,lon,time
    of each MHWS is written to a netcdf file with extension .nc

    Args:
        mhwsys_file (str):
            Output csv file.  Needs to have .csv extension
            netcdf file will match this name but have an _mask.nc extension
        sub (tuple, optional):
            Restrict run to a subset of dates
        cube (numpy.ndarray):
            3D array (lat, lon, day) built from a set of MHWEs
        ymd_start (tuple, optional):
            Defines the first day of the cube
        ignore_hilat (bool, optional):

    """
    if mhwsys_file[-4:] != '.csv':
        raise IOError("Output filename needs to be .csv")
    # Load cube -- See MHW_Cube Notebook
    if cube is None:
        print("Loading cube")
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube.nc'
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data[:].astype(np.int8)
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
    del(cube)

    # maskC[558,857,:]
    print("First pass complete")
    NVoxC = buildc.second_pass(maskC, parentC, catC)
    print("Second pass complete")
    IdToLabel, LabelToId, ndet = utils.prep_labels(maskC, parentC, NVoxC, MinNVox=0, verbose=True)
    obj_dictC = buildc.final_pass(maskC, NVoxC, ndet, IdToLabel, LabelToId, catC)
    print("Objects nearly done")

    # Find maximum Area (km^2)
    buildc.calc_km(maskC, obj_dictC)
    print("area and NVox_km done")

    # Write systems as pandas in HDF
    tbl = utils.dict_to_pandas(obj_dictC, add_latlon=True,
                               start_date=datetime.date(
                                   ymd_start[0], ymd_start[1], 
                                   ymd_start[2]).toordinal())
    tbl.to_csv(mhwsys_file)#, 'mhw_sys', mode='w')
    print("Wrote: {}".format(mhwsys_file))

    # Write mask as nc
    mask_file = mhwsys_file.replace('.csv', '_mask.nc')
    t0 = datetime.datetime(ymd_start[0], ymd_start[1], ymd_start[2])
    times = pd.date_range(start=t0, periods=maskC.shape[2])
    lat_coord, lon_coord = sst_utils.noaa_oi_coords()

    da = xarray.DataArray(maskC, coords=[lat_coord, lon_coord, times],
                          dims=['lat', 'lon', 'time'])
    ds = xarray.Dataset({'mask': da})
    print("Saving mask..")
    encoding = {'mask': dict(compression='gzip',
                         chunksizes=(maskC.shape[0], maskC.shape[1], 1))}
    ds.to_netcdf(mask_file, engine='h5netcdf', encoding=encoding)
    print("Wrote: {}".format(mask_file))

    # Return
    return

# Testing

def main(flg_main):
    # Not sure why this needs to be here.  But it does..
    import xarray

    if flg_main == 'all':
        flg_main = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_main = int(flg_main)
        

    # Test
    if flg_main & (2 ** 0):
        # Scaled seasonalT, thresholdT
        # C testing
        test_c()
        #full_test()

    # Debuggin
    #tbl, mask = main(sub=(11600,11600+380), mhwsys_file='tst.hdf', ymd_start=(2013, 10, 5))
    if flg_main & (2 ** 1):
        cube = np.load('tst_cube.npz')['arr_0'].astype(np.int8)
        # Zero out high/low latitudes
        cube[0:100,:,:] = 0
        cube[-100:,:,:] = 0
        tbl, mask = main(cube=cube, mhwsys_file='tst_systems.hdf', ymd_start=(2013, 10, 5))
        embed(header='134 of build')

    # Testing
    if flg_main & (2 ** 2):
        generate_mhw(sub=(10000,11000))

    # Test Indian
    if flg_main & (2 ** 3):
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
        generate_mhw(cube=cube, mhwsys_file='tst_indian_systems.hdf')

    # Testing
    if flg_main & (2 ** 4):
        import xarray
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc'
        print("Loading: {}".format(cubefile))
        #cubes = iris.load(cubefile) #np.load(cubefile)['cube'].astype(np.int8)
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data[:,:,9500:10000].astype(np.int8)
        print("Loaded!")
        #
        generate_mhw(mhwsys_file='test_basins_systems.hdf', cube=cube)

    # Testing
    if flg_main & (2 ** 5):
        generate_mhw(sub=(10000,11000))

    # Hobday
    if flg_main & (2 ** 6):
        cubefile = os.path.join(mhwdb_path, 'MHWevent_cube_defaults.nc')
        # Cube
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        outfile = os.path.join(mhwdb_path, 'MHWS_defaults.csv')
        generate_mhw(outfile, cube=cube)

    # Vary
    if flg_main & (2 ** 7):
        #import xarray
        #import numpy as np
        #from importlib import reload
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary.nc'
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        generate_mhw(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_vary.csv', cube=cube)

    # Vary 95
    if flg_main & (2 ** 8):
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube_vary_95.nc'
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        generate_mhw(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MHW_systems_vary_95.csv', cube=cube)


    # Cold 10
    if flg_main & (2 ** 9):
        cubefile = '/home/xavier/Projects/Oceanography/MHW/db/MCSevent_cube.nc'
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        generate_mhw(mhwsys_file='/home/xavier/Projects/Oceanography/MHW/db/MCS_systems.csv', cube=cube)

    # 2019, de-trend
    if flg_main & (2 ** 10):
        cubefile = os.path.join(mhwdb_path, 'MHWevent_cube_2019_local.nc')
        # Cube
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        outfile = os.path.join(
            mhwdb_path, 'MHWS_2019_local.csv')
        generate_mhw(outfile, cube=cube)

    # 2019
    if flg_main & (2 ** 11):
        cubefile = os.path.join(
            mhwdb_path, 'MHWevent_cube_2019.nc')
        # Cube
        print("Loading: {}".format(cubefile))
        ds = xarray.open_dataset(cubefile)
        cube = ds.events.data.astype(np.int8)
        ds.close()
        print("Loaded!")
        #
        outfile = os.path.join(
            mhwdb_path, 'MHWS_2019.csv')
        generate_mhw(outfile, cube=cube)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg_main = 0
        flg_main += 2 ** 6  # Hobday MHWEs (1983-2012)
        flg_main += 2 ** 10  # 2019, de-trend
        flg_main += 2 ** 11  # 2019
    else:
        flg_main = sys.argv[1]

    main(flg_main)