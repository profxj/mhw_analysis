""" Module to generate a *large* Cube of MHW events"""
import glob
import numpy as np

import iris

import marineHeatWaves as mhw

from IPython import embed

def load_all_sst(sst_files):
    all_sst = []
    for ifile in sst_files:
        print(ifile)  # For progress
        cubes = iris.load(ifile)
        sst = cubes[0]
        # Get out of lazy
        _ = sst.data
        # Append
        all_sst.append(sst)
    #
    return all_sst

def grab_t(sst_list):
    allts = []
    for sst in sst_list:
        allts += (sst.coord('time').points + 657072).astype(int).tolist()  # 1880?
    return np.array(allts)

def grab_T(sst_list, i, j):
    allTs = []
    for sst in sst_list:
        allTs += sst.data[:,i,j].tolist()
    return np.array(allTs)

def build_me(outfile, noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/', cut=True,
             all_sst=None):
    # Grab the list of SST V2 files
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()

    # Cut?
    if cut:
        all_sst_files = all_sst_files[5:10]

    # Load the Cubes into memory
    if all_sst is None:
        print("Loading up the files. Be patient...")
        all_sst = load_all_sst(all_sst_files)

    # Coords
    lat_coord = all_sst[0].coord('latitude')
    lon_coord = all_sst[0].coord('longitude')
    events_coord = iris.coords.DimCoord(np.arange(100), var_name='events')

    # Time
    t = grab_t(all_sst)

    # Setup for output
    out_dict = {}
    # ints -- all are days
    int_keys = ['time_start', 'time_end', 'time_peak', 'duration', 'duration_moderate', 'duration_strong',
                'duration_severe', 'duration_extreme']
    units = ['day']*len(int_keys)
    for key in int_keys:
        out_dict[key] = np.ma.zeros((lat_coord.shape[0], lon_coord.shape[0], 100), dtype=np.int32, fill_value=-1)

    # Main loop
    if cut:
        irange = range(355, 365)
        jrange = range(715,725)
    else:
        embed(header='72 of build')
    all_mhw = []
    for ilat in irange: #range(355, 365):  # range(lat_coord.shape[0])
        for jlon in jrange: #(715, 725):
            # Temperatures
            SSTs = grab_T(all_sst, ilat, jlon)
            # Detect
            mhws, clim = mhw.detect(t, SSTs, joinAcrossGaps=True)
            # Fill me in
            nevent = mhws['n_events']
            #
            for key in int_keys:
                out_dict[key][ilat, jlon, 0:nevent] = mhws[key]
                out_dict[key][ilat, jlon, nevent:] = np.ma.masked

            # Print me
            print('lat={}, lon={}, nevent={}'.format(lat_coord[ilat].points[0], lon_coord[jlon].points[0],
                                                     mhws['n_events']))
            # Save the dict
            all_mhw.append(mhws)

    # Cubes
    cubes = iris.cube.CubeList()
    for ss, key in enumerate(out_dict.keys()):
        cube = iris.cube.Cube(out_dict[key], units=units[ss], var_name=key,
                                     dim_coords_and_dims=[(lat_coord, 0),
                                                          (lon_coord, 1),
                                                          (events_coord, 2)])
        cubes.append(cube)
    # Write
    iris.save(cubes, outfile, zlib=True)


def dont_run_this():
    # Test from Ipython
    import glob
    from mhw_analysis import build_mhws
    noaa_path='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/'
    all_sst_files = glob.glob(noaa_path + 'sst*nc')
    all_sst_files.sort()
    all_sst_files = all_sst_files[5:10]
    all_sst = build_mhws.load_all_sst(all_sst_files)
    build_mhws.build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.nc', all_sst=all_sst, cut=True)

# Command line execution
if __name__ == '__main__':
    #
    build_me('/home/xavier/Projects/Oceanography/MHWs/test_mhws.nc')

