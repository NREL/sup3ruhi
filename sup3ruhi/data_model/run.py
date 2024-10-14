import os
import shutil
import time
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from rex import Resource, init_logger
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta
from scipy.spatial import KDTree
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.utilities.regridder import Regridder
from sup3r.utilities.utilities import spatial_coarsening

import uhi_data_handling
from uhi_data_handling import (ModisRawLstProduct, EraCity, ModisGfLst,
                               ModisVeg, ModisStaticLayer, ModisAlbedo,
                               GhsData, Nsrdb, Utilities, NetCDF,
                               ATTRS, HR_OBS)


logger = logging.getLogger(__name__)


with open('./source_data/modis_raw/city_bounding_boxes.json', 'r') as f:
    all_cities = json.load(f)


def run(city, city_str, coord, year, pixel_offset, coord_offset, s_enhance,
        lst_fp, veg_fp, topo_fp, land_cover_fp, albedo_fp, era_fp, viewtime_fp,
        ghs_p_fp, ghs_h_fp, ghs_v_fp, nsrdb_fp, fp_out_hr):
    modis_lst = ModisGfLst(lst_fp, coord, coord_offset)
    modis_albedo = ModisAlbedo(albedo_fp)
    era = EraCity(era_fp, coord, coord_offset, pixel_offset, s_enhance)
    nsrdb = Nsrdb(nsrdb_fp, coord, coord_offset)

    target_meta = modis_albedo.meta
    target_shape = modis_albedo.shape
    longitude = modis_albedo.longitude[0, :]
    latitude = modis_albedo.latitude[:, 0]

    mcd = ModisStaticLayer(land_cover_fp, dset='LC_Type1')
    land_cover = mcd.get_data(target_meta=target_meta,
                            target_shape=target_shape)
    land_mask = land_cover != 17

    alb_all = []
    for i in range(1, 8):
        arr = modis_albedo.get_data(f'Nadir_Reflectance_Band{i}',
                                    coarsen=None, land_mask=land_mask)
        alb_all.append(arr)

    ghs_p = GhsData(ghs_p_fp, coord, coord_offset, dset='population', cache=ghs_p_fp.replace('.h5', f'_meta_{city_str}.csv'))
    ghs_h = GhsData(ghs_h_fp, coord, coord_offset, dset='built_height', cache=ghs_h_fp.replace('.h5', f'_meta_{city_str}.csv'))
    ghs_v = GhsData(ghs_v_fp, coord, coord_offset, dset='built_volume', cache=ghs_v_fp.replace('.h5', f'_meta_{city_str}.csv'))

    population = ghs_p.get_data(target_meta, target_shape, mode='sum')
    built_height = ghs_h.get_data(target_meta, target_shape, mode='mean')
    built_volume = ghs_v.get_data(target_meta, target_shape, mode='sum')

    srtm = ModisStaticLayer(topo_fp, dset='SRTMGL3_DEM')
    hr_topo_era = srtm.get_data(era.hr_meta, era.hr_shape)
    topo = srtm.get_data(target_meta, target_shape)

    modis_veg = ModisVeg(veg_fp)

    full_datasets = {'temperature_2m': {},
                     'temperature_max_2m': {},
                     'temperature_min_2m': {},
                     'temperature_mean_2m': {},
                     'u_mean_10m': {},
                     'v_mean_10m': {},
                     'relativehumidity_2m': {},
                     'sea_surface_temperature': {},
                     'lst': {},
                     'evi': {},
                     'albedo_1': {},
                     'albedo_2': {},
                     'albedo_3': {},
                     'albedo_4': {},
                     'albedo_5': {},
                     'albedo_6': {},
                     'albedo_7': {},
                     'ghi': {},
                     'dni': {},
                     'ghi_mean': {},
                     'dni_mean': {},
                     }

    for daynight in ('night', 'day'):
        modis_raw_lst = ModisRawLstProduct(viewtime_fp, daynight)
        evi = modis_veg.get_data(modis_raw_lst.time_index, target_meta, target_shape)
        ghi = nsrdb.get_data('ghi', modis_raw_lst.time_index, target_meta, target_shape)
        dni = nsrdb.get_data('dni', modis_raw_lst.time_index, target_meta, target_shape)
        ghi_mean = nsrdb.get_data('ghi', modis_raw_lst.time_index, target_meta, target_shape, daily_reduce='mean')
        dni_mean = nsrdb.get_data('dni', modis_raw_lst.time_index, target_meta, target_shape, daily_reduce='mean')
        for idt, timestamp in enumerate(modis_raw_lst.time_index):
            doy = timestamp.day_of_year
            era_kws = {'hr_topo': hr_topo_era, 'interpolate': True,
                       'target_meta': target_meta, 'target_shape': target_shape}
            era_trh = era.get_data(['temperature_2m', 'relativehumidity_2m'], timestamp, **era_kws)
            era_t2m = era_trh[..., 0]
            era_rh = era_trh[..., 1]
            era_t2m_max = era.get_data('temperature_2m', timestamp, daily_reduce='max', **era_kws)[..., 0]
            era_t2m_min = era.get_data('temperature_2m', timestamp, daily_reduce='min', **era_kws)[..., 0]
            era_t2m_mean = era.get_data('temperature_2m', timestamp, daily_reduce='mean', **era_kws)[..., 0]
            era_u10m_mean = era.get_data('u_10m', timestamp, daily_reduce='mean', **era_kws)[..., 0]
            era_v10m_mean = era.get_data('v_10m', timestamp, daily_reduce='mean', **era_kws)[..., 0]
            era_sst = era.get_data('sea_surface_temperature', timestamp, **era_kws)[..., 0]
            dn_title = daynight.replace("gh", "").title()
            doy_str = str(timestamp.day_of_year).zfill(3)
            new_lst_fp = os.path.join(os.path.dirname(lst_fp), f'gf_{dn_title}{year}_{doy_str}.h5')

            if os.path.exists(new_lst_fp):
                lst = modis_lst.get_data(era_temp=era_sst, new_file=new_lst_fp, check_coords=False,
                                         target_meta=target_meta, target_shape=target_shape, land_mask=land_mask)

                full_datasets['temperature_2m'][timestamp] = era_t2m
                full_datasets['temperature_max_2m'][timestamp] = era_t2m_max
                full_datasets['temperature_min_2m'][timestamp] = era_t2m_min
                full_datasets['temperature_mean_2m'][timestamp] = era_t2m_mean
                full_datasets['u_mean_10m'][timestamp] = era_u10m_mean
                full_datasets['v_mean_10m'][timestamp] = era_v10m_mean
                full_datasets['relativehumidity_2m'][timestamp] = era_rh
                full_datasets['sea_surface_temperature'][timestamp] = era_sst
                full_datasets['lst'][timestamp] = lst
                full_datasets['evi'][timestamp] = evi[idt]
                full_datasets['albedo_1'][timestamp] = alb_all[0][idt]
                full_datasets['albedo_2'][timestamp] = alb_all[1][idt]
                full_datasets['albedo_3'][timestamp] = alb_all[2][idt]
                full_datasets['albedo_4'][timestamp] = alb_all[3][idt]
                full_datasets['albedo_5'][timestamp] = alb_all[4][idt]
                full_datasets['albedo_6'][timestamp] = alb_all[5][idt]
                full_datasets['albedo_7'][timestamp] = alb_all[6][idt]
                full_datasets['ghi'][timestamp] = ghi[idt]
                full_datasets['dni'][timestamp] = dni[idt]
                full_datasets['ghi_mean'][timestamp] = ghi_mean[idt]
                full_datasets['dni_mean'][timestamp] = dni_mean[idt]

                clear_output(wait=True)
                logger.debug(f'{city} {daynight} {timestamp}')

    darrays = {}
    for key, var_dict in full_datasets.items():
        time_index = pd.to_datetime(sorted(list(var_dict.keys())))
        var_arr = np.dstack([var_dict[ts] for ts in time_index])
        var_arr = np.transpose(var_arr, (2, 0, 1))
        if np.isnan(var_arr).any():
            raise RuntimeError(f'{city} {year} {key} has NaN values!')
        full_datasets[key] = var_arr
        darrays[key] = NetCDF.make_dataarray(var_arr, time_index,
                                             latitude, longitude, ATTRS[key])

    darrays['topography'] = NetCDF.make_dataarray(topo, time_index, latitude, longitude, ATTRS['topography'])
    darrays['land_mask'] = NetCDF.make_dataarray(land_mask, time_index, latitude, longitude, ATTRS['land_mask'])
    darrays['built_volume'] = NetCDF.make_dataarray(built_volume, time_index, latitude, longitude, ATTRS['built_volume'])
    darrays['built_height'] = NetCDF.make_dataarray(built_height, time_index, latitude, longitude, ATTRS['built_height'])
    darrays['population'] = NetCDF.make_dataarray(population, time_index, latitude, longitude, ATTRS['population'])

    encoding = {}
    for var, attrs in ATTRS.items():
        encoding[var] = {k: v for k, v in attrs.items() if k in ('dtype', 'scale_factor')}

    NetCDF.make_dataset(darrays, fp_out=fp_out_hr, encoding=encoding)
    logger.info(f'Wrote file: {fp_out_hr}')


if __name__ == '__main__':

    pixel_offset = 2
    coord_offset = 0.5
    s_enhance = 60
    futures = {}

    init_logger(__name__, log_level='DEBUG')

    test_cities = ['Los Angeles', 'Seattle']

    # Memory bound process, 10 workers with bigmem
    with ProcessPoolExecutor(max_workers=10) as exe:

        #for year in [2018, 2019, 2020]:
            #for city in all_cities:
        for year in [2016, 2017]:
            for city in test_cities:
                city_str = city.lower().replace(" ", "_").replace('.', '').replace('-', '_')

                df = pd.read_csv('./source_data/modis_raw/uscities.csv')
                df = df.sort_values('population', ascending=False)
                df = df.drop_duplicates('city')
                df = df.set_index('city', drop=True)

                coord = df.loc[city, ['lat', 'lng']].values

                lst_fp = f'/scratch/gbuster/modis_lst_gf/modis_lst_gf_{year}/gf_Day{year}_001.h5'
                veg_fp = f'./source_data/modis_raw/downloads/{city_str}_{year}/MYD13A2.061_1km_aid0001.nc'
                topo_fp = f'./source_data/modis_raw/downloads/{city_str}_{year}/SRTMGL3_NC.003_90m_aid0001.nc'
                land_cover_fp = f'./source_data/modis_raw/downloads/{city_str}_{year}/MCD12Q1.061_500m_aid0001.nc'
                albedo_fp = f'./source_data/modis_raw/downloads/{city_str}_{year}/MCD43A4.061_500m_aid0001.nc'
                era_fp = f'./source_data/era_trh/final/era5_trh_{year}.nc'
                viewtime_fp = f'./source_data/modis_raw/downloads/{city_str}_{year}/MYD11A1.061_1km_aid0001.nc'
                ghs_p_fp = './source_data/ghs/ghs_population_100m_e2020.h5'
                ghs_h_fp = './source_data/ghs/ghs_built_height_100m_e2018.h5'
                ghs_v_fp = './source_data/ghs/ghs_built_volume_100m_e2020.h5'

                if year >= 2018:
                    nsrdb_fp = f'/datasets/NSRDB/conus/nsrdb_conus_irradiance_{year}.h5'
                else:
                    nsrdb_fp = f'/datasets/NSRDB/current/nsrdb_{year}.h5'

                fp_out_hr = f'./tmp_data/{city_str}_{year}.nc'

                if not os.path.exists(fp_out_hr):
                    logger.info(f'Starting work on {city} {year}')
                    future = exe.submit(run, city, city_str, coord, year,
                            pixel_offset, coord_offset, s_enhance, lst_fp, veg_fp,
                            topo_fp, land_cover_fp, albedo_fp, era_fp, viewtime_fp,
                            ghs_p_fp, ghs_h_fp, ghs_v_fp, nsrdb_fp, fp_out_hr)
                    futures[future] = city
#                    future = run(city, city_str, coord, year,
#                            pixel_offset, coord_offset, s_enhance, lst_fp, veg_fp,
#                            topo_fp, land_cover_fp, albedo_fp, era_fp,
#                            viewtime_fp, ghs_p_fp, ghs_h_fp, ghs_v_fp, nsrdb_fp,
#                            fp_out_hr)
#
