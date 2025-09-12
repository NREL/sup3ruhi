"""Run the data model to make rasterized training data for the Sup3rUHI model.
The output data is .nc and has solar noon/midnight observations corresponding
with MODIS day/night observations."""

import os
import logging
from concurrent.futures import ProcessPoolExecutor
from rex import init_logger
import pandas as pd
import numpy as np
from IPython.display import clear_output

from sup3ruhi.data_model.data_model import (
    ModisRawLstProduct,
    EraCity,
    ModisGfLst,
    ModisVeg,
    ModisStaticLayer,
    ModisAlbedo,
    GhsData,
    Nsrdb,
    NetCDF,
    ATTRS,
)


PROJ_DIR = '/projects/gates/sup3rcc_uhi/'
SRC_DIR = PROJ_DIR + 'source_data/'
MODIS_DIR = SRC_DIR + 'modis_raw/downloads/'
ERA_FP = SRC_DIR + 'era_trh/final/era5_trh_{year}.nc'
LST_FP = SRC_DIR + 'modis_lst_gf/modis_lst_gf_{year}/gf_Day{year}_001.tif'
GHS_P_FP = SRC_DIR + 'ghs/source/GHS_POP_E2020_*.tif'
GHS_H_FP = SRC_DIR + 'ghs/source/GHS_BUILT_H_ANBH_E2018_*.tif'
GHS_V_FP = SRC_DIR + 'ghs/source/GHS_BUILT_V_E2020_*.tif'
VEG_FP = MODIS_DIR + '{city_str}_{year}/MYD13A2.061_1km_aid0001.nc'
TOPO_FP = MODIS_DIR + '{city_str}_{year}/SRTMGL3_NC.003_90m_aid0001.nc'
LANDC_FP = MODIS_DIR + '{city_str}_{year}/MCD12Q1.061_500m_aid0001.nc'
ALBEDO_FP = MODIS_DIR + '{city_str}_{year}/MCD43A4.061_500m_aid0001.nc'
VIEWTIME_FP = MODIS_DIR + '{city_str}_{year}/MYD11A1.061_1km_aid0001.nc'
NSRDB_FP_2km = '/datasets/NSRDB/conus/nsrdb_conus_irradiance_{year}.h5'
NSRDB_FP_4km = '/datasets/NSRDB/current/nsrdb_{year}.h5'


logger = logging.getLogger(__name__)


def run(
    city,
    coord,
    year,
    pixel_offset,
    coord_offset,
    s_enhance,
    lst_fp,
    veg_fp,
    topo_fp,
    land_cover_fp,
    albedo_fp,
    era_fp,
    viewtime_fp,
    ghs_p_fp,
    ghs_h_fp,
    ghs_v_fp,
    nsrdb_fp,
    fp_out_hr,
):
    """Make a full year of training data in an output .nc file using a mix of
    input files"""
    modis_lst = ModisGfLst(lst_fp, coord, coord_offset)
    modis_albedo = ModisAlbedo(albedo_fp)
    era = EraCity(era_fp, coord, pixel_offset, s_enhance)
    nsrdb = Nsrdb(nsrdb_fp, coord, coord_offset)

    target_meta = modis_albedo.meta
    target_shape = modis_albedo.shape
    longitude = modis_albedo.longitude[0, :]
    latitude = modis_albedo.latitude[:, 0]

    mcd = ModisStaticLayer(land_cover_fp, dset='LC_Type1')
    land_cover = mcd.get_data(
        target_meta=target_meta, target_shape=target_shape
    )
    land_mask = land_cover != 17

    alb_all = []
    for i in range(1, 8):
        arr = modis_albedo.get_data(
            f'Nadir_Reflectance_Band{i}', coarsen=None, land_mask=land_mask
        )
        alb_all.append(arr)

    ghs_p = GhsData(ghs_p_fp, coord, coord_offset)
    ghs_h = GhsData(ghs_h_fp, coord, coord_offset)
    ghs_v = GhsData(ghs_v_fp, coord, coord_offset)

    population = ghs_p.get_data(target_meta, target_shape, mode='sum')
    built_height = ghs_h.get_data(target_meta, target_shape, mode='mean')
    built_volume = ghs_v.get_data(target_meta, target_shape, mode='sum') / 1e3

    srtm = ModisStaticLayer(topo_fp, dset='SRTMGL3_DEM')
    hr_topo_era = srtm.get_data(era.hr_meta, era.hr_shape)
    topo = srtm.get_data(target_meta, target_shape)

    modis_veg = ModisVeg(veg_fp)

    full_datasets = {
        'temperature_2m': {},
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
        evi = modis_veg.get_data(
            modis_raw_lst.time_index, target_meta, target_shape
        )
        ghi = nsrdb.get_data(
            'ghi', modis_raw_lst.time_index, target_meta, target_shape
        )
        dni = nsrdb.get_data(
            'dni', modis_raw_lst.time_index, target_meta, target_shape
        )
        ghi_mean = nsrdb.get_data(
            'ghi',
            modis_raw_lst.time_index,
            target_meta,
            target_shape,
            daily_reduce='mean',
        )
        dni_mean = nsrdb.get_data(
            'dni',
            modis_raw_lst.time_index,
            target_meta,
            target_shape,
            daily_reduce='mean',
        )
        for idt, timestamp in enumerate(modis_raw_lst.time_index):
            era_kws = {
                'hr_topo': hr_topo_era,
                'interpolate': True,
                'target_meta': target_meta,
                'target_shape': target_shape,
            }
            era_trh = era.get_data(
                ['temperature_2m', 'relativehumidity_2m'], timestamp, **era_kws
            )
            era_t2m_max = era.get_data(
                'temperature_2m', timestamp, daily_reduce='max', **era_kws
            )
            era_t2m_min = era.get_data(
                'temperature_2m', timestamp, daily_reduce='min', **era_kws
            )
            era_t2m_mean = era.get_data(
                'temperature_2m', timestamp, daily_reduce='mean', **era_kws
            )
            era_u10m_mean = era.get_data(
                'u_10m', timestamp, daily_reduce='mean', **era_kws
            )
            era_v10m_mean = era.get_data(
                'v_10m', timestamp, daily_reduce='mean', **era_kws
            )
            era_sst = era.get_data(
                'sea_surface_temperature', timestamp, **era_kws
            )

            era_t2m = era_trh[..., 0]
            era_rh = era_trh[..., 1]
            era_t2m_max = era_t2m_max[..., 0]
            era_t2m_min = era_t2m_min[..., 0]
            era_t2m_mean = era_t2m_mean[..., 0]
            era_u10m_mean = era_u10m_mean[..., 0]
            era_v10m_mean = era_v10m_mean[..., 0]
            era_sst = era_sst[..., 0]

            dn_title = daynight.replace('gh', '').title()
            doy_str = str(timestamp.day_of_year).zfill(3)
            new_lst_fp = os.path.join(
                os.path.dirname(lst_fp), f'gf_{dn_title}{year}_{doy_str}.tif'
            )

            if os.path.exists(new_lst_fp):
                new_modis_lst = ModisGfLst(
                    new_lst_fp,
                    coord,
                    coord_offset,
                    yslice=modis_lst.yslice,
                    xslice=modis_lst.xslice,
                )
                lst = new_modis_lst.get_data(
                    era_temp=era_sst,
                    target_meta=target_meta,
                    target_shape=target_shape,
                    land_mask=land_mask,
                )

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
        darrays[key] = NetCDF.make_dataarray(
            var_arr, time_index, latitude, longitude, ATTRS[key]
        )

    darrays['topography'] = NetCDF.make_dataarray(
        topo, time_index, latitude, longitude, ATTRS['topography']
    )
    darrays['land_mask'] = NetCDF.make_dataarray(
        land_mask, time_index, latitude, longitude, ATTRS['land_mask']
    )
    darrays['built_volume'] = NetCDF.make_dataarray(
        built_volume, time_index, latitude, longitude, ATTRS['built_volume']
    )
    darrays['built_height'] = NetCDF.make_dataarray(
        built_height, time_index, latitude, longitude, ATTRS['built_height']
    )
    darrays['population'] = NetCDF.make_dataarray(
        population, time_index, latitude, longitude, ATTRS['population']
    )

    encoding = {}
    for var, attrs in ATTRS.items():
        if var in darrays:
            encoding[var] = {
                k: v
                for k, v in attrs.items()
                if k in ('dtype', 'scale_factor')
            }

    NetCDF.make_dataset(darrays, fp_out=fp_out_hr, encoding=encoding)
    logger.info(f'Wrote file: {fp_out_hr}')
    return fp_out_hr


if __name__ == '__main__':
    pixel_offset = 2
    coord_offset = 0.5
    s_enhance = 60
    futures = {}

    init_logger(__name__, log_level='DEBUG')
    init_logger('sup3ruhi.data_model', log_level='DEBUG')

    test_cities = ['nova']

    # Memory bound process, 10 workers with bigmem
    with ProcessPoolExecutor(max_workers=10) as exe:
        for year in [2018, 2019, 2020]:
            for city in test_cities:
                city_str = (
                    city.lower()
                    .replace(' ', '_')
                    .replace('.', '')
                    .replace('-', '_')
                )

                df = pd.read_csv(SRC_DIR + 'modis_raw/uscities.csv')
                df = df.sort_values('population', ascending=False)
                df = df.drop_duplicates('city')
                df = df.set_index('city', drop=True)

                # coord = df.loc[city, ['lat', 'lng']].values
                coord = (38.980499, -77.445941)  # nova

                lst_fp = LST_FP.format(year=year)
                veg_fp = VEG_FP.format(city_str=city_str, year=year)
                topo_fp = TOPO_FP.format(city_str=city_str, year=year)
                land_cover_fp = LANDC_FP.format(city_str=city_str, year=year)
                albedo_fp = ALBEDO_FP.format(city_str=city_str, year=year)
                era_fp = ERA_FP.format(year=year)
                viewtime_fp = VIEWTIME_FP.format(city_str=city_str, year=year)
                ghs_p_fp = GHS_P_FP
                ghs_h_fp = GHS_H_FP
                ghs_v_fp = GHS_V_FP

                nsrdb_fp = NSRDB_FP_4km.format(year=year)
                if year >= 2018:
                    nsrdb_fp = NSRDB_FP_2km.format(year=year)

                fp_out_hr = PROJ_DIR + f'tmp_data/{city_str}_{year}.nc'

                args = (
                    city,
                    coord,
                    year,
                    pixel_offset,
                    coord_offset,
                    s_enhance,
                    lst_fp,
                    veg_fp,
                    topo_fp,
                    land_cover_fp,
                    albedo_fp,
                    era_fp,
                    viewtime_fp,
                    ghs_p_fp,
                    ghs_h_fp,
                    ghs_v_fp,
                    nsrdb_fp,
                    fp_out_hr,
                )

                if not os.path.exists(fp_out_hr):
                    logger.info(f'Starting work on {city} {year}')
                    # future = exe.submit(run, *args)
                    future = run(*args)
                    futures[future] = city
