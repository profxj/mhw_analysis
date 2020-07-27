""" Build the cube(s) that feeds into build"""

# imports
import numpy as np
from importlib import reload

from datetime import date

import pandas
import sqlalchemy

from IPython import embed

mhw_file = '/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db'

def build_cube(outfile, mhw_events=None):

    # Load event table
    if mhw_events is None:
        engine = sqlalchemy.create_engine('sqlite:///'+mhw_file)
        mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                       columns=['date', 'lon', 'lat', 'duration', 'time_peak',
                                                'ievent', 'time_start', 'index', 'category'])

    print("Events are loaded")

    # Size the cube for coords
    ilon = ((mhw_events['lon'].values - 0.125) / 0.25).astype(np.int32)
    jlat = ((mhw_events['lat'].values + 89.975) / 0.25).astype(np.int32)

    # Times
    min_time = np.min(mhw_events['time_start'])
    #max_time = np.max(mhw_events['time_start'] + mhw_events['duration'])
    ntimes = date(2019, 12, 31).toordinal() - date(1982, 1, 1).toordinal() + 1

    # Categories
    categories = mhw_events['category']

    # Cube me
    cube = np.zeros((720, 1440, ntimes), dtype=np.int8)

    # Do it
    tstart = mhw_events['time_start'].values
    durs = mhw_events['duration'].values

    cube[:] = False
    for kk in range(len(mhw_events)):
        # Convenience
        # iilon, jjlat, tstart, dur = ilon[kk], jlat[kk], time_start[kk], durations[kk]
        #
        if kk % 1000000 == 0:
            print('kk = {}'.format(kk))
        cube[jlat[kk], ilon[kk], tstart[kk]-min_time:tstart[kk]-min_time+durs[kk]] = categories[kk]+1

    # Save
    np.savez_compressed(outfile, cube=cube)
    print("Wrote: {}".format(outfile))

# Testing
if __name__ == '__main__':
    print("Loading the events")
    engine = sqlalchemy.create_engine('sqlite:///' + mhw_file)
    mhw_events = pandas.read_sql_table('MHW_Events', con=engine,
                                       columns=['date', 'lon', 'lat', 'duration', 'time_peak',
                                                'ievent', 'time_start', 'index', 'category'])

    embed(header='67 of cube')
    build_cube('/home/xavier/Projects/Oceanography/MHW/db/MHWevent_cube.npz', mhw_events=mhw_events)
