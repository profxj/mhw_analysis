""" Generate Images for MHW AI analysis"""

import iris


def grab_z500(cube, lon, lat, width):
    """
    Grab Z500 data around the input lon, lat
    with the input width

    Parameters
    ----------
    cube : iris.Cube
    lon : float (deg)
    lat : float (deg)
    width : float (deg)

    Returns
    -------
    numpy.ndarray

    """

    # Generate a constraint
    #  WARNING -- This may not work across the 0-360deg boundary
    constraint = iris.Constraint(latitude=lambda cell: (lat-width/2) <= cell <= (lat+width/2),
                                 longitude=lambda cell: (lon - width/2) <= cell <= (
                                         lon + width/2))
    # Apply
    sub_cube = cube.extract(constraint)

    # Return the data
    return sub_cube.data[:]

