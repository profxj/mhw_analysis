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

from datetime import date
from datetime import datetime
from pandas import DataFrame

import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker

import seaborn	

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


def main():

	# prepare sql stuff 
	mhw_file = '../../Downloads/mhws_allsky_defaults.db'

	engine = sqlalchemy.create_engine('sqlite:///'+mhw_file)
	engine.has_table("MHW_Events")

	connection = engine.connect()

	metadata = sqlalchemy.MetaData()
	mhw_tbl = sqlalchemy.Table('MHW_Events', metadata, autoload=True, autoload_with=engine)

	path = '../Z500'
	zfile = os.path.join(path, 'b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.Z500.18500101-20051231-043.nc')

	# This is slow, so hold in memory
	cubes = iris.load(zfile)

	date = datetime(2000,1,1)
	last = datetime(2000,1,2)

	dmy = (date.day, date.month, date.year)

	any_sst = sst_io.load_noaa(dmy) # This can be any day

	ymd = date.year*10000 + date.month*100 + date.day
	print(ymd)

	reload(cems_io)
	z500_cube = cems_io.load_z500(ymd, cubes)

	z500_noaa = z500_cube.regrid(any_sst, iris.analysis.Linear())

	print(z500_noaa)

	lat = 40.
	lon = 200.
	width = 10.

	lat_start = (lat-width/2)
	lat_end = (lat+width/2)
	lon_start = (lon-width/2)
	lon_end = (lon+width/2)

	constraint = iris.Constraint(latitude=lambda cell: lat_start < cell < lat_end,
	                            longitude = lambda cell: lon_start <= cell <= lon_end)

	print(lat_start)
	print(lat_end)
	print(lon_start)
	print(lon_end)

	# # get z500 data:
	z500_noaa_extract = z500_noaa.extract(constraint)
	z500_lats = z500_noaa_extract.coord('latitude').points
	z500_lons = z500_noaa_extract.coord('longitude').points
	z500_data = z500_noaa_extract.data[:]

	print(z500_lats)
	print(z500_lons)

	# get MHW data:
	range_df = select_range(connection, mhw_tbl, lat_start, lat_end, lon_start, lon_end, date.strftime("%b %d %Y"), date.strftime("%b %d %Y"))
	# # date (12), lat (13), lon (14)
	# print(range_df)

	lats = list(range_df[13])
	lons = list(range_df[14])

	print(lats)
	print(lons)

	print(z500_data.shape)

	plt.clf()
	ax = plt.gca()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# fill_lons = [np.arange(11, 17, 0.5).tolist()]
	
	# a = np.stack([np.arange(35.125, 44.875, 0.25) for _ in range(40)])
	# it = np.arange(35.125, 44.875, 0.25).tolist()
	# print(it)
	# a = []
	# for i, num in enumerate(it):
	# 	for j in range(40):
	# 		a.append(num)

	# b = np.stack([np.arange(195.125, 204.875, 0.25) for _ in range(40)])

	# ax1.scatter(a, b, s=10, c='b', marker="s", label='first')
	ax1.scatter(lats, lons, s=10, c='r', marker="o", label='second')
	plt.legend(loc='upper left')
	plt.xlim(lat_start, lat_end)
	plt.ylim(lon_start, lon_end)
	plt.show()

	# img = ax.pcolormesh(z500_data)
	# # colorbar
	# cb = plt.colorbar(img, fraction=0.030, pad=0.04)

	# seaborn.set(style='ticks')

	# _genders= ['Heatwave']
	# df = pandas.DataFrame({
	#     'latitude': lats,
	#     'longitude': lons,
	#     'type': _genders[0]
	# })

	# fg = seaborn.FacetGrid(data=df, hue='type', hue_order=_genders, aspect=1.61)
	# fg.map(plt.scatter, 'latitude', 'longitude').add_legend()

	# plt.show()

main()