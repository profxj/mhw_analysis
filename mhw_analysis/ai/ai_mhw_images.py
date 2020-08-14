# imports
import os
import numpy as np
from matplotlib import pyplot as plt

import iris

import sys

from dotenv import load_dotenv
import cf_units

from mhw_analysis.cems import io as cems_io
from mhw_analysis.ncep import io as ncep_io
from mhw_analysis.ai import db_utils as db
from mhw_analysis.ai import convert as con

from oceanpy.sst import io as sst_io
from oceanpy.sst import climate as sst_climate

from datetime import timedelta
from datetime import datetime
import dateutil
from pandas import DataFrame

import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker

# import seaborn
import pickle

from skimage.transform import resize


def get_noaa_extract(z500_noaa, bounds):
	lat_start, lat_end, lon_start, lon_end = bounds

	if (lon_start < 0.) or (lon_end > 360.):
		flip = True
		if lon_start < 0.:
			lon_start += 360.
		else:
			tmp = lon_start
			lon_start = lon_end - 360.
			lon_end = tmp
	else:
		flip = False

	# set boundaries
	if flip:
		constraint = iris.Constraint(
		latitude=lambda cell: lat_start < cell < lat_end,
		longitude=lambda cell: (lon_start <= cell) or (cell <= lon_end))
	else:
		constraint = iris.Constraint(
		latitude=lambda cell: lat_start < cell < lat_end,
		longitude=lambda cell: lon_start <= cell <= lon_end)

	# get portion of z500 reframe within these boundaries
	z500_noaa_extract = z500_noaa.extract(constraint)

	return z500_noaa_extract


def show_segmentation_map(lats, lons):
	# ---MAKE PLOTS---

	plt.clf()
	ax = plt.gca()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	l = lat.shape[0]

	a = np.stack([np.arange(np.min(lats), np.max(lats), 0.25) for _ in range(l)])
	it = np.arange(np.min(lats), np.max(lats), 0.25).tolist()
	print(it)
	a = []

	for i, num in enumerate(it):
		for j in range(l):
			a.append(num)

	b = np.stack([np.arange(np.min(lons), np.max(lons), 0.25) for _ in range(l)])

	ax1.scatter(a, b, s=10, c='b', marker="s", label='first')
	ax1.scatter(lats, lons, s=10, c='r', marker="o", label='second')
	plt.legend(loc='upper left')
	# plt.xlim(lat_start, lat_end)
	# plt.ylim(lon_start, lon_end)

	plt.show()

	# ---MAKE PLOTS---

	# ^ re-write this code


def upscale(x, y, new=(256, 256, 1)):
	return resize(x, new), resize(y, new)


def build_X_set(file='../../Downloads/MHW_sys_intermediate.csv', width=16.):
	load_dotenv()

	# load pandas file
	df = pandas.read_csv(file)

	ex_date = datetime(2000, 6, 1)  # compute programatically eventually?

	# get lats, lons, dates from pandas file
	lats, lons, dates = df['lat'], df['lon'], df['date']
	# get curr date
	date = datetime.fromisoformat(dates[0])
	print(date)
	print(type(date))

	# This can be any day
	any_sst = sst_io.load_noaa((ex_date.day, ex_date.month, ex_date.year))

	# get cube data
	sst_ref = con.get_cube_data(any_sst)

	# load cubes and names
	# cubes = cems_io.load_cubes()
	# names = cems_io.load_z500_names(cubes)

	# get all unique dates and store in a hashtable as keys with cubes as values
	set_dates = set(dates)
	uni_dates = {}
	for i, elem in enumerate(set_dates):
		uni_dates[elem] = None

	z500_noaa = None

	lst = []

	# lons 
	# lons = lons.loc[]


	# for i, (lat, lon) in enumerate(zip(lats, lons)):
	lat = -58.
	lon = 356.
	# get the date
	date = ex_date#dates[i]

	# get the bounds
	bounds = get_bounds(lat, lon, width)
	lat_start, lat_end, lon_start, lon_end = bounds

	# if cube not in dict
	# if not uni_dates['']:
	# get curr date
	dt_date = date#datetime.fromisoformat(date)
	print(dt_date)
	# get z500 data for a particular date=date
	z500_noaa = ncep_io.load_z500_noaa_ncep(dt_date, any_sst)
		# uni_dates[date] = z500_noaa
	# else:
	# 	# get from dict
	# 	z500_noaa = uni_dates[date]

	# get the extract for that lat/lon range
	z500_noaa_extract = get_noaa_extract(z500_noaa, bounds)
	z500_data = z500_noaa_extract.data[:]

	# Deal with wrap around
	if (lon_start < 0.) or (lon_end > 360.):
		lons = z500_noaa_extract.coord('longitude')
		# Need to figure out where the split is and
		# repack
		import pdb
		pdb.set_trace()
		print("pdb")

	lst.append(z500_data)

	conv.save_obj(lst, 'X_parts')


def plot(gal=20, pkl='balanced_365_64', path='/Users/kamil/Desktop/research2020/figures'):

	bal = conv.load_obj(pkl)

	for i in range(gal):

		# curr_path = ''
		ind = i

		if i >= (gal/2):
			ind = len(bal) - gal - i

		plt.clf()
		plt.imshow(bal[ind][1])
		plt.gca().invert_yaxis()
		new_path = str(i)+'x'
		plt.savefig(new_path)

		plt.clf()
		ax = plt.gca()
		img = ax.pcolormesh(bal[ind][0])
		# colorbar
		cb = plt.colorbar(img)
		new_path = str(i)+'y'  # os.path.join(curr_path, str(i)+'y')
		plt.savefig(new_path)

# build set
def balance():
	dataset = conv.load_obj('test_365_64_NCEP')
	mhw = []
	for i, elem in enumerate(dataset):
		if np.sum(elem[1]) > 0:
			mhw.append(elem)
			dataset.pop(i)
	np.random.shuffle(dataset)
	mhw = mhw + dataset[:len(mhw)]
	print(len(mhw))
	conv.save_obj(mhw, 'balanced_365_256_NCEP')

# TODO: speed up segmap stuff with spatial locality DONE, re-write plots, multithreading(?)


def main():

	load_dotenv()

	SEG_MAP_DATE = 5
	QUERY_RANGE = 30
	DAYS_IN_SET = 1

	# set start, seg, and end dates
	date = datetime(2000, 6, 1)  # compute programatically eventually?
	seg_date = date + timedelta(days=SEG_MAP_DATE)  # variables
	last = seg_date + timedelta(days=DAYS_IN_SET)  # variables

	# This can be any day
	any_sst = sst_io.load_noaa((date.day, date.month, date.year))

	# modify this to get bigger window
	width = 16.  # 64.#45.#10.
	# amount of lat/lon that increments each "pixel"
	frame_size = .25  # might need to find way to get this programatically

	# modify lats and lons as needed to get more or less data
	true_min_lat = np.min(con.get_cube_lats(any_sst))
	true_min_lon = np.min(con.get_cube_lons(any_sst))
	MIN_LAT = np.min(con.get_cube_lats(any_sst))  # + width
	MIN_LON = np.min(con.get_cube_lons(any_sst))  # + width
	MAX_LAT = np.max(con.get_cube_lats(any_sst))  # - width
	MAX_LON = np.max(con.get_cube_lons(any_sst))  # - width

	# prepare sql stuff, could make it a command line arg
	connection, mhw_tbl = db.get_mhw_tbl()

	# initialize dataset
	dataset = []

	# get cube data
	sst_ref = con.get_cube_data(any_sst)

	# load cubes and names
	cubes = cems_io.load_cubes()
	names = cems_io.load_z500_names(cubes)

	# initialize counter for range_df
	day_count = 0
	range_df = None

	LAT_COL = 1
	LON_COL = 2

	# booleans
	preprocess = False
	NCEP = True

	z500_noaa = None

	# loop over every day
	while seg_date < last:

		if not NCEP:
			# get z500 data for a particular date=date
			z500_noaa = cems_io.load_z500_noaa(date, any_sst, cubes, names)

		else:
			z500_noaa = ncep_io.load_z500_noaa_ncep(date, any_sst)

		if day_count % 30 == 0:
			# get MHW data for a particular date=seg_date, MIGHT NEED TO SPEED THIS UP (CALL METHOD EVERY 6 MONTHS/day, AND ADD MORE CODE TO FILTER DF BY DATE)
			# range_df = db.query_date(connection, mhw_tbl, seg_date.strftime("%b %d %Y"))
			range_df = db.query_date_range(connection, mhw_tbl, seg_date.strftime(
			    "%b %d %Y"), (seg_date + timedelta(days=QUERY_RANGE)).strftime("%b %d %Y"))
			# date (12), lat (13), lon (14)
			print("done with query")
			print(day_count)

		# iterate over every lat and lon frame in that day
		for lat in np.arange(MIN_LAT, MAX_LAT, width):
			for lon in np.arange(330, 360, width):

				bounds = con.get_bounds(lat, lon, width)
				lat_start, lat_end, lon_start, lon_end = bounds

				# if lat_start < MIN_LAT or lon_start < MIN_LON or lat_end > MAX_LAT or lon_end > MAX_LON:
				# 	continue

				# how many values are actually in the ocean in this frame
				# masked = sst_ref[lat_to_row(lat, true_min_lat, frame_size):lat_to_row(lat+width, true_min_lat, frame_size), lon_to_col(lon, true_min_lon, frame_size):lon_to_col(lon+width, true_min_lon, frame_size)].count()
				masked = sst_ref[con.lat_to_row(lat_start, true_min_lat, frame_size):con.lat_to_row(lat_end, true_min_lat, frame_size), con.lon_to_col(
				    lon_start, true_min_lon, frame_size):con.lon_to_col(lon_end, true_min_lon, frame_size)].count()

				# if there are enough values, then check segmentation map
				if masked > 0:

					# create new segmentation map
					# might need to make these + 1
					seg_map = np.zeros((int(width/frame_size), int(width/frame_size)))

					# get the extract for that lat/lon range
					z500_noaa_extract = get_noaa_extract(z500_noaa, bounds)
					z500_data = z500_noaa_extract.data[:]

					# Deal with wrap around
					if (lon_start < 0.) or (lon_end > 360.):
						lons = z500_noaa_extract.coord('longitude')
						# Need to figure out where the split is and
						# repack
						import pdb
						pdb.set_trace()
						print("pdb")

					# get df for that day
					day_df = con.get_day_df(range_df, date.strftime("%b %d %Y"))

					# write pandas code to just get lats, lons from that date
					lats, lons = con.get_lats_lons_from_day(day_df, lat_start, lon_start, width, LAT_COL, LON_COL)
					
					# convert the points to rows/cols
					lats, lons = con.convert_ll_to_rc(np.array(lats), np.array(lons), lat_start, lon_start, frame_size)

					# add heatwave points at those locations
					seg_map[lats[:], lons[:]] = 1

					if preprocess:
						z500_data, seg_map = upscale(z500_data, seg_map)

					print(z500_data.shape)

					# add entry to dataset
					dataset.append((z500_data, seg_map))


		# iterate
		date += timedelta(days=1)
		seg_date += timedelta(days=1)
		day_count += 1
		print(day_count)


	save_name = 'test_'+str(DAYS_IN_SET)+'_'+str(int(width/frame_size))+'_'
	save_name += 'NCEP' if NCEP else ''
	print(save_name)
	# pickle dataset
	# conv.save_obj(dataset, save_name)

main()
# print(conv.load_obj('test'))
# plot()
# balance()

# build_X_set()