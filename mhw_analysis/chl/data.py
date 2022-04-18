""" Module to grab CHL data"""
import subprocess
import os
import numpy as np

from IPython import embed

def grab_cmems(clobber=False):

    for year in np.arange(1993, 2021, dtype=int): 
        for month in np.arange(1, 13, dtype=int): 
            outfile = f'chl_{year}-{month}.nc'
            out_path = os.path.join(os.getenv('CMEMS'), 'CHL') 
            if os.path.isfile(os.path.join(out_path, outfile)) and not clobber:
                print(f"{outfile} exists. Skipping")
                continue
            # 
            print(f"Downloading {outfile}")
            #
            if month < 12:
                date_max = f'{year}-{month+1}-1 00:00:00' 
            else:
                date_max = f'{year}-{month}-31 13:00:00' 
            command = [
                'motuclient',
                '--motu', 
                'https://my.cmems-du.eu/motu-web/Motu',
                '--service-id', 
                'GLOBAL_MULTIYEAR_BGC_001_029-TDS',
                '--product-id', 
                'cmems_mod_glo_bgc_my_0.25_P1D-m', 
                '--longitude-min', '-180', 
                '--longitude-max', '179.75', 
                '--latitude-min','-90', 
                '--latitude-max', '90', 
                '--date-min', f'{year}-{month}-1 00:00:00',
                '--date-max', date_max,
                '--depth-min', '0.5057', 
                '--depth-max', '0.5058',
                '--variable', 'chl',
                '--out-dir', out_path,
                '--out-name', outfile,
                '--user',  'ssakrison', 
                '--pwd', '2Magoosak2',
                ]
            pw = subprocess.Popen(command)
            pw.wait()

if __name__ == '__main__':
    grab_cmems()