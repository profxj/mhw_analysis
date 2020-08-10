""" Utilities for dealing with MHW Events"""
import os
import datetime

import sqlalchemy
import pandas


def db_engine(mhw_file=None):
    # Connect
    if mhw_file is None:
        mhw_file = os.path.join(os.getenv('MHW'), 'db', 'mhws_allsky_defaults.db')
    engine = sqlalchemy.create_engine('sqlite:///' + mhw_file)
    # Return
    return engine

def query_db(engine,
             cols=('date', 'lat', 'lon', 'ievent', 'duration', 'category', 'time_start'),
             latminx=(-90., 90.), lonminx=(0., 360.),
             dateminx=((1990,1,1), (1991,1,1))):


    # Table
    metadata = sqlalchemy.MetaData()
    mhw_tbl = sqlalchemy.Table('MHW_Events', metadata, autoload=True,
                               autoload_with=engine)

    # Build output
    out_cols = []
    for key in cols:
        out_cols += [mhw_tbl.columns[key]]

    # set up query
    query = sqlalchemy.select(out_cols).where(sqlalchemy.and_(
        # range of lat
        sqlalchemy.between(
            # pass functions?
            mhw_tbl.columns.lat, latminx[0], latminx[1]
        ),
        # and range of lon
        sqlalchemy.between(
            # pass functions?
            mhw_tbl.columns.lon, lonminx[0], lonminx[1]
        ),
        # and range of dates
        sqlalchemy.between(
            # pass functions?
            mhw_tbl.columns.date, datetime.datetime(*dateminx[0]), datetime.datetime(*dateminx[1])
        )
    ))

    # Connect
    connection = engine.connect()

    # Execute
    result = connection.execute(query)
    ptbl = pandas.DataFrame(result.fetchall(), columns=cols)

    # Return
    return ptbl
