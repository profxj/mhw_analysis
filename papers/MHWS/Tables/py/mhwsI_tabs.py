#Module for Tables for the MeerTRAP 20201123 paper
# Imports

import numpy as np
import os, sys

import pandas
import xarray

from mhw_analysis.systems import io as mhw_sys_io

from IPython import embed

mhw_sys_file=os.path.join(
    os.getenv('MHW'), 'db', 'MHWS_2019.csv')

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import defs, analy_utils, fitting, analy_sys

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

    # Results files
    # Read
    rpath = '../Analysis/ChangePoint'
    categories = ['severe', 'moderate', 'minor']

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

    for kk, cat in enumerate(categories):
        # Read
        rfile = os.path.join(rpath, f'results_{cat}_ocean_areas.csv')
        df = pandas.read_csv(rfile, index_col=0)

        # Indices
        indices = list(df.index)
        indices.sort()

        tbfil.write('\\multicolumn{5}{c}{'+f'{cat}'+'}\\\\ \n')

        #if kk > 0:
        #    tbfil.write('\\midline\n')

        for index in indices:
            row = df.loc[index]

            # Region
            sline = index 

            # Slope
            try:
                sline += '& {:0.2f}'.format(row.Slope)
            except:
                embed(header='133 of tabs')

            # Slope p-value
            sline += f'& {row["Slope p-value"]:0.2g}'

            # Change point
            year = 1982 + row.Changepoint
            sline += f'& {year}'

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

# Regions
def mktab_regions(outfile='tab_regions.tex'):

    # Open a random SST file
    ds = xarray.open_dataset(os.path.join(os.getenv('NOAA_OI'), 'sst.day.mean.2019.nc'))
    lats = ds.lat.values
    lons = ds.lon.values



    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table}[h]\n')
    tbfil.write('\\begin{center}\n')
    tbfil.write('\\caption{Regions for analysis}')
    tbfil.write('\\label{tab:regions}\n')
    tbfil.write('\\begin{tabular}{ccc}\n')
    tbfil.write('\\topline\n')
    tbfil.write('Region & Latitudes & Longitudes \\\\ \n')
    #tbfil.write('(deg) & (deg) & ($\\arcsec$) & (mag) & classifier & & \\\\ \n')
    tbfil.write('\\midline\n')
    tbfil.write(' \n')

    for region in analy_sys.regions.keys():
        
        # Region
        sline = region

        # Latitudes
        rlats = [lats[min(analy_sys.regions[region]['lat'][ii],719)] for ii in range(2)]
        lbls = ['N' if rlats[ii] > 0 else 'S' for ii in range(2)]

        sline += '&'
        sline += f'{int(abs(rlats[0]))}{lbls[0]}'
        sline += f'--{int(abs(rlats[1]))}{lbls[1]}'

        # Longitudes
        rlons = [lons[min(analy_sys.regions[region]['lon'][ii],1439)] for ii in range(2)]
        rlons = [rlons[ii] if rlons[ii] < 180 else rlons[ii]-360 for ii in range(2)]
        lbls = ['E' if rlons[ii] > 0 else 'W' for ii in range(2)]

        sline += '&'
        sline += f'{int(abs(rlons[0]))}{lbls[0]}'
        sline += f'--{int(abs(rlons[1]))}{lbls[1]}'

        # Finish
        tbfil.write(sline + '\\\\ \n')

    # End end
    tbfil.write('\\botline \n')
    tbfil.write('\\end{tabular}\n')
    tbfil.write('\\end{center}\n')
    tbfil.write('\\end{table}\n')

    tbfil.close()
    print('Wrote {:s}'.format(outfile))


def mhws_percentile():
    # Load
    mhw_sys = mhw_sys_io.load_systems(
        mhw_sys_file=mhw_sys_file)

    ntot = len(mhw_sys)

    for cat in [defs.classa, defs.classb, defs.classc]:
        gd_mhws = (mhw_sys.NVox_km > defs.type_dict_km[cat][0]) & (
            mhw_sys.NVox_km <= defs.type_dict_km[cat][1])
        #
        nsub = np.sum(gd_mhws)
        print(f"We have {nsub} systems with category {cat}")
        print(f"{nsub/ntot}%")

#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

# Command line execution
if __name__ == '__main__':

    #mktab_mhws(sub=True)
    #mktab_change()
    #mhws_percentile()
    mktab_regions()