import glob, os 

from IPython.terminal.embed import embed
import numpy as np
import scipy.ndimage as ndimage
from datetime import date

import xarray


istart=None
iend=None
def load_chl(chl_path:str, chl_root:str,
                 climatologyPeriod:tuple):
    """
    Load up all the chl data for a series of years

    Args:
        chl_path (str):
        chl_root (str):
        climatologyPeriod (tuple): starting year, end year [int]
        interpolated (bool, optional):
            Use interpolated SST files?

    Returns:
        tuple:  lat_coord, lon_coord, np.array of toordials, list of masked SST

    """
    # Grab the list of SST V1 files
    all_chl_files = glob.glob(os.path.join(chl_path, chl_root))
    all_chl_files.sort()

    # Load the Cubes into memory
    for ii, ifile in enumerate(all_chl_files):
        if str(climatologyPeriod[0]) in ifile:
            global istart
            istart = ii
        if str(climatologyPeriod[-1]) in ifile:
            global iend
            iend = ii+1
    chl_files = all_chl_files[istart:iend]

    print("Loading up the files. Be patient...")
    all_chl = []
    allts = []
    for kk, ifile in enumerate(chl_files):
        print(ifile)  # For progress
        ds = xarray.open_dataset(ifile)
        # lat, lon
        if kk == 0:
            lat = ds.lat 
            lon = ds.lon
    
        datetimes = ds.time.values.astype('datetime64[s]').tolist()
        t = [datetime.toordinal() for datetime in datetimes]
        chl = ds.CHL.to_masked_array()
        # Append 
        all_chl.append(chl)
        allts += t
    #
    return lat, lon, np.array(allts), all_chl


# Command line execution
if __name__ == '__main__':
    load_chl()