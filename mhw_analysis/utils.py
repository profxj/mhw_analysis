import datetime

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


def grab_geo_subimg(da, lats, lons):
    pass