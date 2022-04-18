import glob, os 

from IPython.terminal.embed import embed
import numpy as np

import xarray


def load_chl(chl_path:str, chl_root:str,
                 climatologyPeriod:tuple):
    """
    Load up all the chl data for a series of years

    THIS IS ONLY SETUP FOR CMEMS

    Args:
        chl_path (str):
        chl_root (str):
        climatologyPeriod (tuple): starting year, end year [int]

    Returns:
        tuple:  lat_coord, lon_coord, np.array of toordials, list of masked SST

    """
    istart=None
    iend=None

    # Grab the list of SST V1 files
    all_chl_files = glob.glob(os.path.join(chl_path, chl_root))
    all_chl_files.sort()

    # Load the Cubes into memory
    for ii, ifile in enumerate(all_chl_files):
        if str(climatologyPeriod[0]) in ifile and istart is None:
            istart = ii
        if str(climatologyPeriod[-1]) in ifile:
            iend = ii+1

    chl_files = all_chl_files[istart:iend]

    # Now order by time!
    t0s = []
    for ifile in chl_files:
        ds = xarray.open_dataset(ifile)
        t0 = ds.time.values[0]
        t0s.append(t0)
    srt = np.argsort(t0s)
    chl_files = np.array(chl_files)[srt].tolist()

    print("Loading up the files. Be patient...")
    all_chl = []
    allts = []
    for kk, ifile in enumerate(chl_files):
        print(ifile)  # For progress
        ds = xarray.open_dataset(ifile)
        # lat, lon
        if kk == 0:
            lat = ds.latitude
            lon = ds.longitude
    
        datetimes = ds.time.values.astype('datetime64[s]').tolist()
        t = [datetime.toordinal() for datetime in datetimes]
        chl = ds.chl.to_masked_array()

        # Append 
        all_chl.append(chl[:,0,:,:])
        allts += t
    #
    return lat, lon, np.array(allts), all_chl


# Command line execution
if __name__ == '__main__':
    load_chl()