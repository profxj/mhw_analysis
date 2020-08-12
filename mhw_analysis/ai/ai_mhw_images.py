# imports
from importlib import reload
import os
import numpy as np
from matplotlib import pyplot as plt

import iris

import sys

# create module paths
# MHW_PATH = '/Users/kamil/Desktop/research2020/mhw_analysis'
# OCEANPY_PATH = '/Users/kamil/Desktop/research2020/oceanpy'
# os.environ['NOAA_OI'] = '/Users/kamil/Desktop/NOAA-OI-SST-V2'

# append paths
# sys.path.append(MHW_PATH)
# sys.path.append(OCEANPY_PATH)


from mhw_analysis.cems import io as cems_io
from oceanpy.sst import io as sst_io

from datetime import timedelta
# from datetime import date
from datetime import datetime
from pandas import DataFrame

import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker

# import seaborn	
import pickle

def select_range(connection, table, lat_start=-89.875, lat_end=89.875, lon_start=.125, lon_end=355.875, date_start='Jun 1 2005', date_end='Jun 1 2006'):
  """
  Queries MHW_Events table given ranges (latitude, longitude, date), and returns
  a dataframe containing all the entries of the table which meet that criteria.
  If no range is given, will give all MHW events from that date. 

  Query (SQL):
  SELECT * FROM MHW_Events 
  WHERE lat BETWEEN ? AND ? 
  AND lon BETWEEN ? AND ?
  AND date BETWEEN ? AND ?

  :param connection:
    # connect to engine
    connection = engine.connect()
    SQLConnection to the actual engine so that the engine doesn't need to be passed around

  :param table:
    SQLAlchemy Table object containing MHW_Events
    Run the following code before calling this function:
      engine = sqlalchemy.create_engine('sqlite:///'+mhw_file)
      metadata = sqlalchemy.MetaData()
      mhw_tbl = sqlalchemy.Table('MHW_Events', metadata, autoload=True, autoload_with=engine)

  :param lat_start:
    double representing the lower end of latitude range
  
  :param lat_end:
    double representing the upper end of latitude range
  
  :param lon_start:
    double representing the lower end of longitude range
  
  :param lon_end:
    double representing the lower end of longitude range
  
  :param date_start:
    String representing the start date
    Format: 'Mon Day Year' where Day and Year are integers
  
  :param date_end:
    String representing the start date
    Format: 'Mon Day Year' where Day and Year are integers
  
  :return range_df:
    Pandas dataframe containing query results
  """

  # can't run if table not properly set up
  # if not table:
  #   print("Please load the MHW_Events table into a Table object.")
  #   return None
  
  # set up query
  query = sqlalchemy.select([table]).where(sqlalchemy.and_(
    # range of lat 
    sqlalchemy.between(
        # pass functions?
        table.columns.lat, lat_start, lat_end
    ),
    # and range of lon
    sqlalchemy.between(
        # pass functions?
        table.columns.lon, lon_start, lon_end
    ),
    # and range of dates
    sqlalchemy.between(
        # pass functions?
        table.columns.date, datetime.strptime(date_start, '%b %d %Y'), datetime.strptime(date_end, '%b %d %Y')
    )
    ))
  
  # execute query
  result = connection.execute(query)

  # get df
  range_df = DataFrame(result.fetchall())

  # range_df.columns = query.keys()

  # return df
  return range_df

def query_date(connection, table, date='Jun 1 2005'):
  # set up query
  # SELECT lat, lon FROM MHW_Events
  # WHERE date == date
  query = sqlalchemy.select([table.columns.lat, table.columns.lon]).where(table.columns.date == datetime.strptime(date, '%b %d %Y'))

  # execute query
  result = connection.execute(query)

  # print(result)

  # get df
  df = DataFrame(result.fetchall())
  # range_df.columns = query.keys()

  # return df
  return df

# used to save the tuples 
# could include folder path as a param
def save_obj(obj, name, folder_path='mhw_data/'):
    # save object to folder
    with open(folder_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, folder_path='mhw_data/'):
    # load object from folder
    with open(folder_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_mhw_tbl(mhw_file='../../Downloads/mhws_allsky_defaults.db'):
    # create engine and check if it has the table we need
    engine = sqlalchemy.create_engine('sqlite:///'+mhw_file)
    engine.has_table("MHW_Events")

    # connect to engine
    connection = engine.connect()

    # load table from metadata
    metadata = sqlalchemy.MetaData()
    mhw_tbl = sqlalchemy.Table('MHW_Events', metadata, autoload=True, autoload_with=engine)

    # return connection and table
    return connection, mhw_tbl

def load_z500_noaa(date, any_sst, path='../Z500', document='b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.Z500.18500101-20051231-043.nc'):
    # convert date to proper format
    ymd = date.year*10000 + date.month*100 + date.day

    # create filepath
    zfile = os.path.join(path, document)

    # This is slow, so hold in memory
    cubes = iris.load(zfile)

    # get cube at ymd date
    z500_cube = cems_io.load_z500(ymd, cubes)

    # get z500 regridded data
    z500_noaa = z500_cube.regrid(any_sst, iris.analysis.Linear())

    return z500_noaa

def get_noaa_extract(z500_noaa, bounds):
    lat_start, lat_end, lon_start, lon_end = bounds

    if (lon_start < 0.) or (lon_end > 360.):
        flip = True
        if lon_start < 0.:
            lon_start += 360.
        else:
            pass
            #pass
            #tmp = lon_start
            #lon_start = lon_end - 360.
            #lon_end = tmp
    else:
        flip = False

    # set boundaries
    if flip:
        constraint = iris.Constraint(
            latitude=lambda cell: lat_start < cell < lat_end,
            longitude = lambda cell: (lon_start <= cell) or (cell <= lon_end))
    else:
        constraint = iris.Constraint(
            latitude=lambda cell: lat_start < cell < lat_end,
            longitude = lambda cell: lon_start <= cell <= lon_end)

    # get portion of z500 reframe within these boundaries
    z500_noaa_extract = z500_noaa.extract(constraint)

    return z500_noaa_extract

def show_segmentation_map(lats, lons):
    # ---MAKE PLOTS---

    plt.clf()
    ax = plt.gca()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    a = np.stack([np.arange(35.125, 44.875, 0.25) for _ in range(40)])
    it = np.arange(35.125, 44.875, 0.25).tolist()
    print(it)
    a = []
    for i, num in enumerate(it):
        for j in range(40):
            a.append(num)

    b = np.stack([np.arange(195.125, 204.875, 0.25) for _ in range(40)])

    ax1.scatter(a, b, s=10, c='b', marker="s", label='first')
    ax1.scatter(lats, lons, s=10, c='r', marker="o", label='second')
    plt.legend(loc='upper left')
    # plt.xlim(lat_start, lat_end)
    # plt.ylim(lon_start, lon_end)

    plt.show()

    # ---MAKE PLOTS---

    # ^ re-write this code

# getter for latitudes of any iris cube
def get_cube_lats(cube):
    return cube.coord('latitude').points

# getter for latitudes of any iris cube
def get_cube_lons(cube):
    return cube.coord('longitude').points

# getter for data of any iris cube
def get_cube_data(cube):
    return cube.data[:]

def lat_to_row(lat, true_min_lat, frame_size):
    return int((lat - true_min_lat)/(frame_size))

def lon_to_col(lon, true_min_lon, frame_size):
    return int((lon - true_min_lon)/(frame_size))

# method to convert lats and lons (as numpy arrays) to row/cols
def convert_ll_to_rc(lats, lons, lat_start, lon_start, frame_size):
    lats = ((lats - lat_start)/(frame_size)).astype(int)
    lons = ((lons - lon_start)/(frame_size)).astype(int)
    return lats, lons

def get_lats_lons_from_day(df, lat, lon, width, lat_col=13, lon_col=14):
    df = df[(df[lat_col] >= lat) & (df[lat_col] < (lat+width)) & (df[lon_col] >= lon) & (df[lon_col] < (lon+width))]
    return df[lat_col], df[lon_col]

def get_bounds(lat, lon, width):
    # create boundaries, make separate method when u start iterating over different frames
    lat_start = (lat-width/2)
    lat_end = (lat+width/2)
    lon_start = (lon-width/2)
    lon_end = (lon+width/2)

    return [lat_start, lat_end, lon_start, lon_end]

# TODO: speed up segmap stuff with spatial locality, re-write plots, multithreading(?)
def main():
    from dotenv import load_dotenv
    load_dotenv()

    SEG_MAP_DATE = 5
    DAYS_IN_SET = 365

    # set start, seg, and end dates
    date = datetime(2000,6,1) # compute programatically eventually?
    seg_date = date + timedelta(days=SEG_MAP_DATE) # variables
    last = seg_date + timedelta(days=DAYS_IN_SET) # variables

    # This can be any day
    any_sst = sst_io.load_noaa((date.day, date.month, date.year))

    # modify this to get bigger window
    width = 10.
    # amount of lat/lon that increments each "pixel"
    frame_size = .25 # might need to find way to get this programatically

    # modify lats and lons as needed to get more or less data
    true_min_lat = np.min(get_cube_lats(any_sst))
    true_min_lon = np.min(get_cube_lons(any_sst))
    MIN_LAT = np.min(get_cube_lats(any_sst)) #+ width
    MIN_LON = np.min(get_cube_lons(any_sst)) #+ width
    MAX_LAT = np.max(get_cube_lats(any_sst)) #- width
    MAX_LON = np.max(get_cube_lons(any_sst)) #- width

    # prepare sql stuff, could make it a command line arg
    connection, mhw_tbl = get_mhw_tbl()

    # initialize dataset
    dataset = []

    # get lats at [lat_to_row(lat),lat_to_row(lat+10)] and lons at [lon_to_col(lon), lon_to_col(lon+10)]
    # get cube data
    sst_ref = get_cube_data(any_sst)

    # loop over every day
    while seg_date < last:

        # get z500 data for a particular date=date
        z500_noaa = load_z500_noaa(date, any_sst)

        # get MHW data for a particular date=seg_date, MIGHT NEED TO SPEED THIS UP (CALL METHOD EVERY 6 MONTHS/day, AND ADD MORE CODE TO FILTER DF BY DATE)
        range_df = query_date(connection, mhw_tbl, seg_date.strftime("%b %d %Y"))
        # date (12), lat (13), lon (14)

        # iterate over every lat and lon frame in that day
        for lat in np.arange(MIN_LAT, MAX_LAT, width):
            for lon in np.arange(MIN_LON, MAX_LON, width):

                bounds = get_bounds(lat, lon, width)
                lat_start, lat_end, lon_start, lon_end = bounds

                #if lat_start < MIN_LAT or lon_start < MIN_LON or lat_end > MAX_LAT or lon_end > MAX_LON:
                #	continue

                # how many values are actually in the ocean in this frame
                # masked = sst_ref[lat_to_row(lat, true_min_lat, frame_size):lat_to_row(lat+width, true_min_lat, frame_size), lon_to_col(lon, true_min_lon, frame_size):lon_to_col(lon+width, true_min_lon, frame_size)].count()
                masked = sst_ref[lat_to_row(lat_start, true_min_lat, frame_size):lat_to_row(lat_end, true_min_lat, frame_size), lon_to_col(lon_start, true_min_lon, frame_size):lon_to_col(lon_end, true_min_lon, frame_size)].count()

                print(masked)

                # if there are enough values, then check segmentation map
                if masked > 0:

                    # create new segmentation map
                    seg_map = np.zeros((int(width/frame_size),int(width/frame_size))) # might need to make these + 1

                    # get the extract for that lat/lon range
                    z500_noaa_extract = get_noaa_extract(z500_noaa, bounds)
                    z500_data = z500_noaa_extract.data[:]

                    # Deal with wrap around
                    if (lon_start<0.) or (lon_end>360.):
                        lons = z500_noaa_extract.coord('longitude')
                        # Need to figure out where the split is and
                        # repack
                        import pdb; pdb.set_trace()

                    # if there are any MHWs on this day,
                    # if range_df is not None and not range_df.empty:

                    # write pandas code to just get lats, lons from that date
                    lats, lons = get_lats_lons_from_day(range_df, lat_start, lon_start, width, 0, 1)


                    # print(np.min(lats))
                    # print(np.max(lats))
                    # print(lat_start)
                    # print((lat_start+width))

                    # print("range_df")

                    # convert the points to rows/cols
                    lats, lons = convert_ll_to_rc(np.array(lats), np.array(lons), lat_start, lon_start, frame_size)

                    # print(lats)
                    # print(lons)

                    # add heatwave points at those locations
                    seg_map[lats[:], lons[:]] = 1

                    # add entry to dataset
                    dataset.append((z500_data, seg_map))
                    # print("FINISHED")

        # iterate
        date += timedelta(days=1)
        seg_date += timedelta(days=1)

        # pickle dataset
        save_obj(dataset, 'test')

    # main()
    print(load_obj('test'))

# make method to modularize getting the z500 data DONE
# set the start of the 5 days in advance DONE
# make method for lat_start, lat_end ... DONE
# optimize conversion to rows/cols with numpy operations, convert to method? DONE
# test on one day DONE
# write code to introduce the seg map for the 5 days in advance DONE
# test on one day DONE
# write code to loop it DONE
# test on one week DONE
# write code to loop lat and lon frames
# create 1 year dataset