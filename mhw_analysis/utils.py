import datetime

import xarray

def doy(time):
    """
    Calculate the day of year from input datetime object

    Args:
        time (datetime.datetime):

    Returns:
        int: day of year where January 1 has a value of 1 not 0

    """
    day_of_year = (time - datetime.datetime(time.year, 1, 1)).days + 1
    # Leap year?
    if (time.year % 4) != 0:
        if day_of_year <= 28:
            pass
        else:
            day_of_year += 1
    #
    return day_of_year


def grab_geo_subimg(da, lats, lons, fix_coord=False):
    """

    Parameters
    ----------
    da
    lats
    lons
    fix_coord

    Returns
    -------

    """
    # Allow for negative lons input
    if lons[0] < 0:
        lons[0] += 360.

    if lons[0] > lons[1]:
        # West
        subda_A = da.sel(lon=slice(lons[0], 360.),
                        lat=slice(lats[0], lats[1]))
        if fix_coord:
            subda_A = subda_A.assign_coords(lon=(((subda_A.lon + 180) % 360) - 180))
        # East
        subda_B = da.sel(lon=slice(0., lons[1]),
                        lat=slice(lats[0], lats[1]))
        # Concatenate
        sub_da = xarray.concat([subda_A, subda_B], dim='lon')
    else:
        sub_da = da.sel(lon=slice(lons[0], lons[1]),
                        lat=slice(lats[0], lats[1]))
    # Return
    return sub_da