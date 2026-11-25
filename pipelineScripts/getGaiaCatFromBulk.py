import os
import sys
import glob
import pandas as pd
import concurrent.futures
import threading 
from datetime import datetime
from astropy import units as u
from astropy_healpix import HEALPix
from astroquery.gaia import Gaia
import numpy as np
from astropy.io import fits
from astropy.table import Table,vstack
import numpy as np

def getSourcesCSV(ra,dec,sizeOfField,spectraDir):
    DR3=True
    target_table='gaia_source'
    hpx_level = 6
    lon=ra*u.deg
    lat=dec*u.deg
    radius=(sizeOfField/2)*u.deg

    output_dir=spectraDir
    output_file=output_dir+"/gaia_tabs.txt"
    if not os.path.isdir(f'{output_dir}'):
        os.system(f'mkdir {output_dir}')
    gaia_dr_flag='DR3'
    url_prefix      = f'http://cdn.gea.esac.esa.int/Gaia/g{gaia_dr_flag.lower()}/{target_table}/'
    md5sum_file_url = url_prefix + '_MD5SUM.txt'
    md5sum_file     = pd.read_csv(md5sum_file_url, header=None, delim_whitespace=True, names=['md5Sum', 'file'])
    md5sum_file.drop(md5sum_file.tail(1).index,inplace=True)
    # Extract HEALPix level-8 from file name ======================================
    healpix_8_min  = [int(file[file.find('_')+1:file.rfind('-')])     for file in md5sum_file['file']]
    healpix_8_max  = [int(file[file.rfind('-')+1:file.rfind('.csv')]) for file in md5sum_file['file']]
    reference_file = pd.DataFrame({'file':md5sum_file['file'], 'healpix8_min':healpix_8_min, 'healpix8_max':healpix_8_max}).reset_index(drop=True)

    # Compute HEALPix levels 6,7, and 9 ===========================================
    reference_file['healpix7_min'] = [inp >> 2 for inp in reference_file['healpix8_min']]
    reference_file['healpix7_max'] = [inp >> 2 for inp in reference_file['healpix8_max']]

    reference_file['healpix6_min'] = [inp >> 2 for inp in reference_file['healpix7_min']]
    reference_file['healpix6_max'] = [inp >> 2 for inp in reference_file['healpix7_max']]

    reference_file['healpix9_min'] = [inp << 2       for inp in reference_file['healpix8_min']]
    reference_file['healpix9_max'] = [(inp << 2) + 3 for inp in reference_file['healpix8_max']]

    # Generate reference file =====================================================
    ncols          = ['file', 'healpix6_min', 'healpix6_max', 'healpix7_min', 'healpix7_max', 'healpix8_min', 'healpix8_max', 'healpix9_min', 'healpix9_max']
    reference_file = reference_file[ncols]
    hp             = HEALPix(nside=2**hpx_level, order='nested')
    hp_cone_search = hp.cone_search_lonlat(lon, lat, radius = radius) 

    f = open(output_file,"w")
    subset=[]
    for index in reference_file.index:
        row = reference_file.iloc[index]
        hp_min, hp_max = row[f'healpix{hpx_level}_min'], row[f'healpix{hpx_level}_max']
        if np.any(np.logical_and(hp_min <= hp_cone_search, hp_cone_search <= hp_max)):
            bulk_file = url_prefix + row['file'] + '\n'
            f.write(bulk_file)
            subset.append(bulk_file)
    f.close()
    os.system(f'wget -i {output_file} -P {output_dir}/ -q  --show-progress --progress=bar:force 2>&1')
    os.system(f'gunzip {output_dir}/*.gz')
    return()

def parse_gaia_array(s):
    # 1. s[1:-1] elimina el '[' del inicio y el ']' del final
    # 2. sep=',' le dice a numpy que corte por comas
    return np.fromstring(s[1:-1], sep=',')

def generateCatalog(csvDir,ra,dec,sizeOfField):
    raMin = ra - sizeOfField / 2
    raMax = ra + sizeOfField / 2
    decMin = dec - sizeOfField / 2
    decMax = dec + sizeOfField / 2
    #catalog=Table(names=['ra','dec','ra_error','dec_error','phot_g_mean_mag','phot_g_mean_flux','phot_g_mean_flux_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error'],dtype=['f8']*13)
    columns_to_keep = ['ra','dec','ra_error','dec_error','phot_g_mean_mag','phot_g_mean_flux','phot_g_mean_flux_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error']
    tables = []
    for file in glob.glob(f"{csvDir}/*.csv"):
        df = pd.read_csv(file, sep=',',skiprows=1000)
        df_filtered = df[
            (df['ra']  >= raMin)  & (df['ra']  <= raMax) &
            (df['dec'] >= decMin) & (df['dec'] <= decMax)
        ]
        if not df_filtered.empty:
            # Keep only existing columns to avoid KeyError
            existing = [col for col in columns_to_keep if col in df_filtered.columns]
            tables.append(Table.from_pandas(df_filtered[existing]))

    if tables:
        return vstack(tables)
    else:
        return Table(names=columns_to_keep, dtype=['f8'] * len(columns_to_keep))

def saveCatalog(catalog,outputFile):
    catalog.write(outputFile, format='fits', overwrite=True)
    return

csvDir = sys.argv[1]
ra = float(sys.argv[2])
dec = float(sys.argv[3])
sizeOfField = float(sys.argv[4])
outputFile = sys.argv[5]

getSourcesCSV(ra,dec,sizeOfField,csvDir)
catalog = generateCatalog(csvDir,ra,dec,sizeOfField)
saveCatalog(catalog,outputFile)