import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from datetime import timedelta
from datetime import datetime
import dateutil
from pandas import DataFrame

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
      mhw_tbl = sqlalchemy.Table(
          'MHW_Events', metadata, autoload=True, autoload_with=engine)

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
        table.columns.date, datetime.strptime(
            date_start, '%b %d %Y'), datetime.strptime(date_end, '%b %d %Y')
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
  query = sqlalchemy.select([table.columns.lat, table.columns.lon]).where(
      table.columns.date == datetime.strptime(date, '%b %d %Y'))

  # execute query
  result = connection.execute(query)

  # print(result)

  # get df
  df = DataFrame(result.fetchall())
  # range_df.columns = query.keys()

  # return df
  return df


def query_date_range(connection, table, date_start='Jun 1 2005', date_end='Jun 2 2005'):
  # set up query
  # SELECT lat, lon FROM MHW_Events
  # WHERE date BETWEEN date_start, date_end
  query = sqlalchemy.select([table.columns.date, table.columns.lat, table.columns.lon]).where(
  	# and range of dates
    sqlalchemy.between(
        # pass functions?
        table.columns.date, datetime.strptime(
            date_start, '%b %d %Y'), datetime.strptime(date_end, '%b %d %Y')
    ))

  # execute query
  result = connection.execute(query)

  # get df
  df = DataFrame(result.fetchall())

  # return df
  return df

def get_mhw_tbl(mhw_file='../../Downloads/mhws_allsky_defaults.db'):
  # create engine and check if it has the table we need
  engine = sqlalchemy.create_engine('sqlite:///'+mhw_file)
  engine.has_table("MHW_Events")

  # connect to engine
  connection = engine.connect()

  # load table from metadata
  metadata = sqlalchemy.MetaData()
  mhw_tbl = sqlalchemy.Table('MHW_Events', metadata,
                             autoload=True, autoload_with=engine)

  # return connection and table
  return connection, mhw_tbl