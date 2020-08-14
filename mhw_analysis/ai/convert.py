import iris
import pandas

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
	df = df[(df[lat_col] >= lat) & (df[lat_col] < (lat+width)) &
	         (df[lon_col] >= lon) & (df[lon_col] < (lon+width))]
	return df[lat_col], df[lon_col]


def get_bounds(lat, lon, width):
	# create boundaries, make separate method when u start iterating over different frames
	lat_start = (lat-width/2)
	lat_end = (lat+width/2)
	lon_start = (lon-width/2)
	lon_end = (lon+width/2)

	return [lat_start, lat_end, lon_start, lon_end]


def get_day_df(df, date='Jun 1 2005', date_col=0):
	return df.loc[df[date_col] == date]

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