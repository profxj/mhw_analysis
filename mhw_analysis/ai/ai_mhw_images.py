# imports
from importlib import reload
import os
import numpy as np
from matplotlib import pyplot as plt

import iris

import sys

# create module paths
MHW_PATH = '/Users/kamil/Desktop/research2020/mhw_analysis'
OCEANPY_PATH = '/Users/kamil/Desktop/research2020/oceanpy'
os.environ['NOAA_OI'] = '/Users/kamil/Desktop/NOAA-OI-SST-V2'

# append paths
sys.path.append(MHW_PATH)
sys.path.append(OCEANPY_PATH)

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

def select_range(connection, table, lat_start=30.0, lat_end=35.0, lon_start=30.0, lon_end=35.0, date_start='Jun 1 2005', date_end='Jun 1 2006'):
  """
  Queries MHW_Events table given ranges (latitude, longitude, date), and returns
  a dataframe containing all the entries of the table which meet that criteria. 

  Query (SQL):
  SELECT * FROM MHW_Events 
  WHERE lat BETWEEN ? AND ? 
  AND lon BETWEEN ? AND ?
  AND date BETWEEN ? AND ?

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

# used to save the tuples 
# could include folder path as a param
def save_obj(obj, name):
	# save object to folder
    with open('mhw_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    # load object from folder
    with open('mhw_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_mhw_tbl(mhw_file):
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

def load_z500_noaa(path, date, any_sst):
	# convert date to proper format
	ymd = date.year*10000 + date.month*100 + date.day

	# create filepath
	zfile = os.path.join(path, 'b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.Z500.18500101-20051231-043.nc')

	# This is slow, so hold in memory
	cubes = iris.load(zfile)

	# get cube at ymd date
	reload(cems_io)
	z500_cube = cems_io.load_z500(ymd, cubes)

	# get z500 regridded data
	z500_noaa = z500_cube.regrid(any_sst, iris.analysis.Linear())

	return z500_noaa

def get_noaa_extract(z500_noaa, lat, lon, width):
	# create boundaries, make separate method when u start iterating over different frames
	lat_start = (lat-width/2)
	lat_end = (lat+width/2)
	lon_start = (lon-width/2)
	lon_end = (lon+width/2)

	# set boundaries
	constraint = iris.Constraint(latitude=lambda cell: lat_start < cell < lat_end,
	                            longitude = lambda cell: lon_start <= cell <= lon_end)

	# get portion of z500 reframe within these boundaries
	z500_noaa_extract = z500_noaa.extract(constraint)

	return z500_noaa_extract, lat_start, lat_end, lon_start, lon_end

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

def get_cube_lats(cube):
	return cube.coord('latitude').points

def get_cube_lons(cube):
	return cube.coord('longitude').points
	
def get_cube_data(cube):
	return cube.data[:]

# TODO: modularize lat/lon => row/col conversion, iterate lat/lon, speed up segmap stuff with spatial locality, re-write plots, multithreading(?)
def main():

	# prepare sql stuff, could make it a command line arg
	mhw_file = '../../Downloads/mhws_allsky_defaults.db'
	connection, mhw_tbl = get_mhw_tbl(mhw_file)

	# prep z500 path
	z500_path = '../Z500'

	# set start, seg, and end dates
	date = datetime(2000,1,1) # compute programatically eventually?
	seg_date = date + timedelta(days=5) # variables
	last = seg_date + timedelta(days=2) # variables

	width = 10. # modify this to get bigger window

	any_sst = sst_io.load_noaa((date.day, date.month, date.year)) # This can be any day

	# initialize dataset
	dataset = []

	while seg_date < last:

		# start loop 
		dmy = (date.day, date.month, date.year)

		# arbitrary, these will need to be modified and iterated somehow
		lat = 40. 
		lon = 200.

		# get z500 data for a particular date=date
		z500_noaa = load_z500_noaa(z500_path, date, any_sst)
		z500_noaa_extract, lat_start, lat_end, lon_start, lon_end = get_noaa_extract(z500_noaa, lat, lon, width)
		z500_data = z500_noaa_extract.data[:]

		# get MHW data, MIGHT NEED TO SPEED THIS UP (CALL METHOD EVERY 6 MONTHS, AND ADD MORE CODE TO FILTER DF BY DATE)
		range_df = select_range(connection, mhw_tbl, lat_start, lat_end, lon_start, lon_end, seg_date.strftime("%b %d %Y"), seg_date.strftime("%b %d %Y"))
		# # date (12), lat (13), lon (14)

		# ---CONVERT LATS/LONS TO ROWS/COLS---
		
		# amount of lat/lon that increments each "pixel"
		frame_size = .25 # might need to find way to get this programatically

		# create new segmentation map
		seg_map = np.zeros((int(width/frame_size),int(width/frame_size))) # might need to make these + 1

		if range_df is not None:
			lats = np.array(range_df[13])
			lons = np.array(range_df[14])
			# data = np.ones(len(lats)) # data = list(range_df[??])

			lats = ((lats - lat_start)/(frame_size)).astype(int)
			lons = ((lons - lon_start)/(frame_size)).astype(int)
			
			# add heatwave points
			seg_map[lats[:], lons[:]] = 1


		# add entry to dataset
		dataset.append((z500_data, seg_map))

		# iterate
		date += timedelta(days=1)
		seg_date += timedelta(days=1)

	# pickle dataset
	save_obj(dataset, 'test')

main()
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