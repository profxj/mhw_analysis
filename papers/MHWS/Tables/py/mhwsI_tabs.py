#Module for Tables for the MeerTRAP 20201123 paper
# Imports

import numpy as np
import os, sys

import pandas

from mhw_analysis.systems import io as mhw_sys_io

from IPython import embed

mhw_sys_file=os.path.join(
    os.getenv('MHW'), 'db', 'MHWS_2019.csv')

# MHWS
def mktab_mhws(outfile='tab_mhws.tex', sub=False):

    if sub:
        outfile = 'tab_mhws_sub.tex'

    # Load
    mhw_sys = mhw_sys_io.load_systems(
        mhw_sys_file=mhw_sys_file)

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table}[h]\n')
    tbfil.write('\\begin{center}\n')
    tbfil.write('\\caption{Marine Heat Wave Systems}')
    tbfil.write('\\label{tab:mhws}\n')
    tbfil.write('\\begin{tabular}{ccccccc}\n')
    tbfil.write('\\topline\n')
    tbfil.write('ID & Lat & Lon & Start & \\tdur & \\maxa & \\nvox \\\\ \n')
    tbfil.write(' & (deg) & (deg) & & (days) & (\\aunit) & (days \\aunit)  \\\\ \n')
    #tbfil.write('(deg) & (deg) & ($\\arcsec$) & (mag) & classifier & & \\\\ \n')
    tbfil.write('\\midline\n')
    tbfil.write(' \n')

    for ss in range(len(mhw_sys)):
        if sub and ss > 25:
            break
        mhws = mhw_sys.iloc[ss]

        # ID
        sline = f'{mhws.mask_Id}'

        # Lat
        sline += '& {:0.3f}'.format(mhws.lat)

        # Lon
        sline += '& {:0.3f}'.format(mhws.lon)

        # Start
        sline += f'& {str(mhws.startdate)[0:10]}'

        # Duration
        sline += f'& {mhws.duration.days}'

        # Max Area
        sline += f'& {mhws.max_area_km:0.3g}'

        # NVox
        sline += f'& {mhws.NVox_km:0.3g}'

        # Finish
        tbfil.write(sline + '\\\\ \n')

    # End end
    tbfil.write('\\botline \n')
    tbfil.write('\\end{tabular}\n')
    tbfil.write('\\end{center}\n')
    tbfil.write('\\end{table}\n')

    tbfil.close()
    print('Wrote {:s}'.format(outfile))

# Changepoint
def mktab_change(outfile='tab_changepoint.tex'):

    # Read
    df = pandas.read_csv('../Analysis/results_ocean_areas.csv')

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table}[h]\n')
    tbfil.write('\\begin{center}\n')
    tbfil.write('\\caption{Change point analysis}')
    tbfil.write('\\label{tab:change}\n')
    tbfil.write('\\begin{tabular}{ccccc}\n')
    tbfil.write('\\topline\n')
    tbfil.write('Region & Slope & p-value & Changepoint	& p-value \\\\ \n')
    #tbfil.write('(deg) & (deg) & ($\\arcsec$) & (mag) & classifier & & \\\\ \n')
    tbfil.write('\\midline\n')
    tbfil.write(' \n')

    for ss in range(len(df)):
        row = df.iloc[ss]

        # Region
        sline = f'{row.Region}'

        # Slope
        sline += '& {:0.2f}'.format(row.Slope)

        # Slope p-value
        sline += f'& {row["Slope p-value"]:0.2g}'

        # Change point
        sline += f'& {row.Changepoint}'

        # Change point p-value
        sline += f'& {row["Changepoint p-value"]:0.2g}'

        # Finish
        tbfil.write(sline + '\\\\ \n')

    # End end
    tbfil.write('\\botline \n')
    tbfil.write('\\end{tabular}\n')
    tbfil.write('\\end{center}\n')
    tbfil.write('\\end{table}\n')

    tbfil.close()
    print('Wrote {:s}'.format(outfile))



#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

# Command line execution
if __name__ == '__main__':

    mktab_mhws(sub=True)
    #mktab_change()