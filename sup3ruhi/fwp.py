import os
import calendar
import datetime
import time
import logging
import numpy as np
import pandas as pd
from glob import glob
from inspect import signature
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import pearsonr
import xarray as xr
import plotly.express as px
from warnings import warn

from rex import init_logger
from phygnn import TfModel
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan as GanModel
from sup3r.models.surface import SurfaceSpatialMetModel
from sup3r.utilities.regridder import Regridder
from sup3r.preprocessing.dual_batch_handling import DualBatchHandler
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC, DualDataHandler
from sup3r.utilities.utilities import spatial_coarsening, transform_rotate_wind

from sup3ruhi.data_model.data_model import EraCity, ModisStaticLayer, Nsrdb, Utilities

logger = logging.getLogger(__name__)


class Sup3rUHI:

    CITY_VARS = ('sea_surface_temperature',
                 'evi',
                 'albedo_1',
                 'albedo_2',
                 'albedo_3',
                 'albedo_4',
                 'albedo_5',
                 'albedo_6',
                 'albedo_7',
                 'built_volume',
                 'built_height',
                 'population',
                 'topography',
                 'land_mask')
    """City attributes that are assumed not to vary year to year"""

    def __init__(self, data_fp_trh,
                 data_fp_city,
                 data_fp_solar,
                 model_fp_lst,
                 model_fp_trh,
                 year,
                 hr_obs,
                 months=None,
                 era_reanalysis=True,
                 model_trh_pad=7,
                 coord_offset=0.5,
                 pixel_offset=2,
                 s_enhance=60,
                 dsets_trh=('temperature_2m', 'relativehumidity_2m'),
                 use_cpu=True,
                 max_workers=1,
                 ):
        """
        Parameters
        ----------
        data_fp_trh : str
            ERA .nc filepath with one year of ~31km hourly data. Dataset shape
            is expected to be (time, lat, lon). Valid 3D surface datasets:
            temperature_2m, relativehumidity_2m, u_10m, v_10m
        data_fp_city : str
            Filepath to Sup3rUHI .nc file for a single city. Typical resolution
            is 500m with 2x obs per day (~12hr timesteps). This is typically
            created by running the Sup3rUHI code to create spatiotemporal data
            on city characteristics from many data sources. Dataset shape is
            expected to be (time, latitude, longitude). Valid datasets include:
            ['sea_surface_temperature', 'evi', 'albedo_1', 'albedo_2',
             'albedo_3', 'albedo_4', 'albedo_5', 'albedo_6', 'albedo_7',
             'topography', 'land_mask', 'built_volume', 'built_height',
             'population']
        model_fp_lst : str
            Directory path to a Sup3rGan model trained to predict LST based on
            many high-resolution city features (from data_fp_city). This is a
            fully convolutional network without any spatiotemporal enhancement
            that operates on 5D arrays of shape (obs, lat, lon, time, features)
        model_fp_trh : str
            Directory path to a phygnn TfModel trained to predict 2-meter air
            temperature and relative humidity based on many high-resolution
            city features (from data_fp_city). This is a fully convolutional
            network that operates on 4D arrays of shape
            (time, lat, lon, features)
        max_workers : int | None
            Number of parallel works to use to perform interpolation of ERA
            temp/humidity data. 1 is serial, None is all available.
        """

        self.lst = None
        self.t2m = None
        self.rh2m = None
        self.trh_pred = None
        self.model_trh_pad = model_trh_pad
        self.coord_offset = coord_offset
        self.pixel_offset = pixel_offset
        self.year = year
        self.months = months
        self.s_enhance = s_enhance
        self.dsets_trh = dsets_trh
        self.max_workers = max_workers
        self._u10m = None
        self._v10m = None
        if self.months is None:
            self.months = tuple(range(1, 13))

        if use_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.model_lst = GanModel.load(model_fp_lst)
        self.model_trh = TfModel.load(model_fp_trh)

        self.ti_12hr, self.tslice_12hr = self.get_tslice(self.year,
                                                         self.months,
                                                         '12h',
                                                         hr_obs=hr_obs,
                                                         drop_leap=True)
        self.ti_1hr, self.tslice_1hr = self.get_tslice(self.year,
                                                       self.months,
                                                       '1h',
                                                       drop_leap=True)

        logger.debug('Loading city data...')
        self.city_data = DataHandlerNC(
            file_paths=data_fp_city,
            features=self.CITY_VARS,
            lr_only_features=self.CITY_VARS,
            hr_exo_features=tuple(),
            time_chunk_size=int(1e6),
            val_split=0.0,
            hr_spatial_coarsen=1,
            temporal_slice=self.tslice_12hr,
            worker_kwargs=dict(max_workers=1))

        self.coord = (self.city_data.meta['latitude'].mean(),
                      self.city_data.meta['longitude'].mean())

        self.solar, self.ghi, self.dni = self.get_solar(data_fp_solar)

        if era_reanalysis:
            self.get_trh_era(data_fp_trh)
        else:
            self.get_trh_h5(data_fp_trh)

        logger.info('Finished initializing Sup3rUHI object!')

    @staticmethod
    def _find_index(name_list, tag):
        """Find the entry in the name list that has the tag"""
        out = None
        for idf, name in enumerate(name_list):
            if tag in name:
                out = idf
                break
        return out

    @property
    def lat_lon(self):
        """Get the coordinates associated with the input/output raster. Shape
        is (n_lat, n_lon, 2) where lat_lon[..., 0] is latitude and
        lat_lon[..., 1] is longitude"""
        return self.city_data.lat_lon

    def get_trh_era(self, data_fp_trh):
        """Get hourly temperature and humidity data from ERA5 and interpolate
        to high-res city meta data."""
        self.trh_dh = EraCity(data_fp_trh, self.coord, self.coord_offset,
                              self.pixel_offset, self.s_enhance)

        idf_topo = self.city_data.features.index('topography')
        hr_topo = self.city_data.data[:, :, 0, idf_topo]
        hr_topo = np.expand_dims(hr_topo.flatten(), -1)
        regridder = Regridder(self.city_data.meta, self.trh_dh.hr_meta)
        hr_topo_era = regridder(hr_topo).reshape(self.trh_dh.hr_shape)

        logger.debug('Loading ERA t/rh data...')
        target_era_shape = self.city_data.shape[:2] + (len(self.ti_1hr),)
        self.trh_data = self.trh_dh.get_data(self.dsets_trh,
                                             time=self.tslice_1hr,
                                             hr_topo=hr_topo_era,
                                             interpolate=True,
                                             daily_reduce=None,
                                             target_meta=self.city_data.meta,
                                             target_shape=target_era_shape,
                                             max_workers=self.max_workers)

    def get_trh_h5(self, data_fp_trh):
        """Get hourly temperature and humidity data from an NREL h5 (e.g.,
        nsrdb or sup3rcc) and interpolate to high-res city meta data."""
        logger.debug('Loading H5 t/rh data...')
        ny = self.city_data.lat_lon.shape[0] // self.s_enhance
        nx = self.city_data.lat_lon.shape[1] // self.s_enhance
        ny = self.s_enhance * ny
        nx = self.s_enhance * nx
        lr_lat_lon = spatial_coarsening(self.city_data.lat_lon[:ny, :nx],
                                        obs_axis=False,
                                        s_enhance=self.s_enhance)
        lr_meta = pd.DataFrame({'latitude': lr_lat_lon[..., 0].flatten(),
                                'longitude': lr_lat_lon[..., 1].flatten()})
        self.trh_dh = Nsrdb(data_fp_trh, self.coord, self.coord_offset)

        self.trh_data = []
        for dset in self.dsets_trh:
            if (dset.startswith(('u_', 'v_'))
                    and 'dset' not in self.trh_dh.handle):
                arr = self.get_uv_from_wsd(dset, lr_lat_lon, lr_meta)
            else:
                arr = self.trh_dh.get_data(dset, self.ti_1hr, lr_meta,
                                           lr_lat_lon.shape[:2],
                                           daily_reduce=None)

            arr = np.expand_dims(arr, -1)
            self.trh_data.append(arr)

        self.trh_data = np.concatenate(self.trh_data, -1)

        idf_topo = self.city_data.features.index('topography')
        hr_topo = self.city_data.data[:ny, :nx, 0, idf_topo]

        model = SurfaceSpatialMetModel(self.dsets_trh, self.s_enhance)
        lr_topo = spatial_coarsening(np.expand_dims(hr_topo, -1),
                                     s_enhance=self.s_enhance,
                                     obs_axis=False)[..., 0]
        exo_data = [{'data': lr_topo}, {'data': hr_topo}]
        exo_data = {'topography': {'steps': exo_data}}
        # SurfaceSpatialMetModel requires 4D (obs, space, space, features)
        self.trh_data = model.generate(self.trh_data,
                                       exogenous_data=exo_data)
        ny = self.city_data.shape[0] - self.trh_data.shape[1]
        nx = self.city_data.shape[1] - self.trh_data.shape[2]
        if ny > 0 or nx > 0:
            padding = ((0, 0), (0, nx), (0, ny), (0, 0))
            self.trh_data = np.pad(self.trh_data, padding, mode='edge')

    def get_uv_from_wsd(self, dset, lr_lat_lon, lr_meta):
        """Get u/v wind components from windspeed/direction from sup3rcc"""

        assert 'windspeed_10m' in self.trh_dh.handle
        assert 'winddirection_10m' in self.trh_dh.handle

        if self._u10m is None and self._v10m is None:
            ws = self.trh_dh.get_data('windspeed_10m', self.ti_1hr,
                                      lr_meta, lr_lat_lon.shape[:2],
                                      daily_reduce=None)
            wd = self.trh_dh.get_data('winddirection_10m', self.ti_1hr,
                                      lr_meta, lr_lat_lon.shape[:2],
                                      daily_reduce=None)
            # ws/wd will be in (time, lat, lon).
            # transform_rotate_wind wants (lat, lon, time)
            ws = np.transpose(ws, (1, 2, 0))
            wd = np.transpose(wd, (1, 2, 0))
            u10m, v10m = transform_rotate_wind(ws, wd, lr_lat_lon)
            self._u10m = np.transpose(u10m, (2, 0, 1))
            self._v10m = np.transpose(v10m, (2, 0, 1))

        if dset.startswith('u_'):
            return self._u10m
        else:
            return self._v10m

    @staticmethod
    def get_tslice(year, months, freq, hr_obs=None, drop_leap=True):
        """Get a subset time index and a slice to slice the full time index
        for the requested months

        Note that historical city data had 12/31 dropped in leap years because
        no LST data for IAState and we haven't had a leap year after the LST
        data availability yet, so drop 12/31 everywhere for now
        """

        if hr_obs is None:
            ti = pd.date_range(f'{year}0101', f'{year+1}0101',
                               freq=freq, inclusive='left')
        else:
            ti = Utilities.get_generic_ti(year, np.min(hr_obs), np.max(hr_obs))

        if calendar.isleap(year) and drop_leap:
            mask = (ti.month == 12) & (ti.day == 31)
            ti = ti[~mask]

        tslice = np.where(ti.month.isin(months))[0]
        tslice = slice(tslice[0], tslice[-1] + 1)
        ti = pd.to_datetime(ti[tslice])
        return ti, tslice

    def get_solar(self, data_fp_solar):
        """Get solar data"""
        solar = Nsrdb(data_fp_solar, self.coord, self.coord_offset)
        logger.debug('Loading solar data...')
        ghi = solar.get_data('ghi', self.ti_1hr, self.city_data.meta,
                             self.city_data.shape[:2],
                             daily_reduce=None)
        dni = solar.get_data('dni', self.ti_1hr, self.city_data.meta,
                             self.city_data.shape[:2],
                             daily_reduce=None)
        return solar, ghi, dni

    def daily_reduce(self, arr, feature):
        tslices = []
        for tstamp in self.ti_12hr:
            mask = tstamp.date() == self.ti_1hr.date
            tslice = np.where(mask)[0]
            tslice = slice(tslice[0], tslice[-1]+1)
            tslices.append(tslice)

        if '_max' in feature.casefold():
            out = np.dstack([arr[tslice].max(0) for tslice in tslices])
        elif '_min' in feature.casefold():
            out = np.dstack([arr[tslice].min(0) for tslice in tslices])
        elif '_mean' in feature.casefold():
            out = np.dstack([arr[tslice].mean(0) for tslice in tslices])

        out = np.transpose(out, (2, 0, 1))

        return out

    def make_lst_input(self, yslice=slice(None), xslice=slice(None)):
        """Make spatiotemporal data inputs for LST model"""

        logger.debug('Making LST input...')

        lst_input = []
        tslice = np.where(self.ti_1hr.isin(self.ti_12hr))[0]

        for idf, feature in enumerate(self.model_lst.lr_features):

            base_name = feature.replace('_max', '').replace('_min', '')
            base_name = base_name.replace('_mean', '')

            if feature in self.city_data.features:
                idf_source = self.city_data.features.index(feature)
                arr = self.city_data.data[..., idf_source]
                arr = np.transpose(arr, (2, 0, 1))
                arr = arr[:, yslice, xslice]

            elif feature == 'ghi_mean':
                arr = self.daily_reduce(self.ghi[:, yslice, xslice], feature)

            elif feature == 'dni_mean':
                arr = self.daily_reduce(self.dni[:, yslice, xslice], feature)

            elif feature == 'ghi':
                arr = self.ghi[tslice, yslice, xslice]

            elif feature == 'dni':
                arr = self.dni[tslice, yslice, xslice]

            elif base_name in self.dsets_trh and feature in self.dsets_trh:
                idf_source = self.dsets_trh.index(base_name)
                arr = self.trh_data[tslice, yslice, xslice, idf_source]

            elif base_name in self.dsets_trh and feature not in self.dsets_trh:
                idf_source = self.dsets_trh.index(base_name)
                arr = self.trh_data[:, yslice, xslice, idf_source]
                arr = self.daily_reduce(arr, feature)

            else:
                raise

            arr = np.expand_dims(arr, -1)
            lst_input.append(arr)

        lst_input = np.concatenate(lst_input, -1)
        self.lst_input = lst_input
        # (time, lat, lon, features)

        logger.debug('Finished making LST input.')

        return lst_input

    def _interp_trh_input_arr(self, arr):
        """
        arr must be (time, space, space)
        """
        ti_12hr_int = self.ti_12hr.values.astype(int)
        ti_1hr_int = self.ti_1hr.values.astype(int)

        flat_shape = [arr.shape[0], np.prod(arr.shape[1:])]
        out_shape = [len(self.ti_1hr), *arr.shape[1:]]

        interp_in = arr.reshape(flat_shape)
        interp_out = np.zeros((len(ti_1hr_int), flat_shape[1]), dtype=np.float32) * np.nan

        for idx in range(flat_shape[1]):
            fill_value = (interp_in[0, idx], interp_in[-1, idx])
            interp = interp1d(ti_12hr_int, interp_in[:, idx],
                              bounds_error=False, fill_value=fill_value)
            interp_out[:, idx] = interp(ti_1hr_int)

        interp_out = interp_out.reshape(out_shape)

        return interp_out

    def _pad_trh_input_arr(self, arr):
        # TRH model crops spatial footprint, pad here to have good output size
        padding = ((0, 0), (self.model_trh_pad, self.model_trh_pad),
                   (self.model_trh_pad, self.model_trh_pad))
        arr = np.pad(arr, padding, mode='edge')
        return arr

    def make_trh_input(self, lst=None, yslice=slice(None), xslice=slice(None)):
        """Make spatiotemporal data inputs for T2M model"""

        logger.debug('Making TRH input...')

        if lst is None and self.lst is None:
            logger.debug('LST input to T2M model is None, '
                         'running LST model...')
            lst = self.generate_lst(lst_input=None, yslice=yslice, xslice=xslice)
        elif lst is None and self.lst is not None:
            logger.debug('LST input to T2M model is None, '
                         'using self.lst...')
            lst = self.lst

        lst_input = self.lst_input
        if self.lst_input is None:
            lst_input = self.make_lst_input(yslice=yslice, xslice=xslice)

        # (time, lat, lon, features)
        shape = [len(self.ti_1hr),
                 self.city_data.shape[0] + 2 * self.model_trh_pad,
                 self.city_data.shape[1] + 2 * self.model_trh_pad,
                 len(self.model_trh.feature_names)]
        if yslice is not None and yslice != slice(None):
            shape[1] = len(np.arange(shape[1])[yslice])
            shape[1] += 2 * self.model_trh_pad
        if xslice is not None and xslice != slice(None):
            shape[2] = len(np.arange(shape[2])[xslice])
            shape[2] += 2 * self.model_trh_pad
        trh_input = np.zeros(shape, dtype=np.float32)

        for idf, feature in enumerate(self.model_trh.feature_names):
            roll = 0
            if '_roll' in feature:
                roll = feature.split('_')[-1]
                assert roll.startswith('roll')
                feature = feature.replace(f'_{roll}', '')
                roll = int(roll.strip('roll'))

            if feature in self.dsets_trh:
                idf_source = self.dsets_trh.index(feature)
                # idt_source = np.where(self.ti_1hr.isin(self.ti_12hr))[0]
                arr = self.trh_data[:, yslice, xslice, idf_source]

            elif feature == 'ghi':
                arr = self.ghi[:, yslice, xslice]
            elif feature == 'dni':
                arr = self.dni[:, yslice, xslice]

            elif feature == 'lst':
                arr = self._interp_trh_input_arr(lst)

            elif feature in self.model_lst.lr_features:
                idf_source = self.model_lst.lr_features.index(feature)
                arr = lst_input[..., idf_source]
                arr = self._interp_trh_input_arr(arr)

            if roll > 0:
                arr = np.roll(arr, roll, axis=0)

            trh_input[..., idf] = self._pad_trh_input_arr(arr)

        self.trh_input = trh_input

        logger.debug('Finished making TRH input.')

        return trh_input

    def generate_lst(self, lst_input=None, chunks=None, yslice=slice(None), xslice=slice(None)):
        """Generate LST data"""

        if lst_input is None:
            logger.debug('Input to LST model is None, creating input data...')
            lst_input = self.make_lst_input(yslice=yslice, xslice=xslice)

        logger.debug('Running LST model...')
        # lst_input is created as (time, lat, lon, feature) but LST
        # model wants (1, lat, lon, time, feature)
        lst_input = np.transpose(lst_input, (1, 2, 0, 3))
        lst_input = np.expand_dims(lst_input, 0)

        if chunks is None:
            lst = self.model_lst.generate(lst_input,
                                          norm_in=True,
                                          un_norm_out=True)
        else:
            input_chunks = np.array_split(lst_input, chunks, axis=3)
            lst = [self.model_lst.generate(ix, norm_in=True, un_norm_out=True)
                   for ix in input_chunks]
            lst = np.concatenate(lst, axis=3)

        logger.debug('Finished LST model!')

        self.lst = np.transpose(lst[0, ..., 0], (2, 0, 1))
        # (time, lat, lon)
        return self.lst

    def _rescale_trh_profiles(self, model_pred, feature_tag,
                              yslice=slice(None), xslice=slice(None)):
        """Use the UHI model predictions to rescale the hourly TRH profiles

        Parameters
        ----------
        model_pred : np.ndarray
            Air temperature and relative humidity model output in shape
            (time, lat, lon, features) where time is usually the 2x per day
            frequency corresponding to satellite observations.
        feature_tag : str
            Tag (substring) of feature name being processed e.g., temperature
            or humidity
        """

        ti_1hr, _ = self.get_tslice(self.city_data.time_index.year[0],
                                    self.months, '1h', drop_leap=True)

        idf_era = self._find_index(self.dsets_trh, feature_tag)
        profiles = self.trh_data[:, yslice, xslice, idf_era]
        df_hourly = pd.DataFrame(profiles.reshape((len(profiles), -1)),
                                 index=ti_1hr)
        df_hourly = df_hourly.round(2)

        idf_pred = self._find_index(self.model_trh.label_names, feature_tag)
        df_pred = model_pred[..., idf_pred]
        df_pred = df_pred.reshape((len(model_pred), -1))
        df_pred = pd.DataFrame(df_pred,
                               index=self.city_data.time_index.round('1h'))
        df_pred = df_pred.reindex(df_hourly.index)

        scalar = df_pred / df_hourly
        scalar = scalar.clip(lower=-0.1, upper=3)  # prevent inf
        scalar = scalar.interpolate('linear', axis=0).bfill().ffill()

        df_pred = df_hourly * scalar
        shape = (len(df_pred), int(df_pred.shape[1]**0.5),
                 int(df_pred.shape[1]**0.5))
        out = df_pred.values.reshape(shape)

        return out

    def generate_trh(self, trh_input=None, chunks=None, yslice=slice(None), xslice=slice(None)):
        """Generate T2M data"""
        if trh_input is None:
            logger.debug('Input to TRH model is None, '
                         'making TRH model input...')
            trh_input = self.make_trh_input(lst=None, yslice=yslice, xslice=xslice)

        logger.debug('Running TRH model...')

        if chunks is None:
            self.trh_pred = self.model_trh.predict(trh_input)
        else:
            out_shape = (trh_input.shape[0],
                         trh_input.shape[1] - 2 * self.model_trh_pad,
                         trh_input.shape[2] - 2 * self.model_trh_pad,
                         len(self.model_trh.label_names))
            self.trh_pred = np.zeros(out_shape, dtype=np.float32) * np.nan
            run_chunks = np.array_split(np.arange(len(trh_input)), chunks)
            run_slices = [slice(id0[0], id0[-1]+1) for id0 in run_chunks]
            for idc, run_slice in enumerate(run_slices):
                iout = self.model_trh.predict(trh_input[run_slice])
                self.trh_pred[run_slice] = iout
                logger.debug(f'Finished chunk {idc+1} out of {len(run_chunks)}')

        if np.isnan(self.trh_pred).sum() > 0:
            n_nan = np.isnan(self.trh_pred).sum()
            n_tot = self.trh_pred.size
            perc = 100 * n_nan / n_tot
            msg = (f'TRH output prediction had {n_nan} NaNs '
                   f'out of {n_tot} ({perc:.2f}%)')
            warn(msg)
            logger.warning(msg)

        idf_t2m = self._find_index(self.model_trh.label_names, 'temperature')
        idf_rh2m = self._find_index(self.model_trh.label_names, 'humidity')
        self.t2m = self.trh_pred[..., idf_t2m]
        self.rh2m = self.trh_pred[..., idf_rh2m]
        self.rh2m = np.maximum(self.rh2m, 0)
        self.rh2m = np.minimum(self.rh2m, 100)
        logger.debug('Finished TRH model!')

        return self.t2m, self.rh2m

    def generate(self, chunks=None, yslice=slice(None), xslice=slice(None)):
        lst_input = self.make_lst_input(yslice=yslice, xslice=xslice)
        lst = self.generate_lst(lst_input, chunks=chunks, yslice=yslice, xslice=xslice)
        trh_input = self.make_trh_input(lst, yslice=yslice, xslice=xslice)
        t2m, rh2m = self.generate_trh(trh_input, chunks=chunks, yslice=yslice, xslice=xslice)
        return t2m, rh2m, lst
