"""Classes to handle various data formats for fusion in Sup3rUHI"""
import os
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor
from cftime import date2num
import datetime
from dateutil import parser
from rex import Resource, MultiFileResource
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import xarray as xr
import logging
from datetime import timedelta
import warnings

from sup3r.models.surface import SurfaceSpatialMetModel
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.utilities.utilities import nn_fill_array, spatial_coarsening
from sup3r.utilities.regridder import Regridder


logger = logging.getLogger(__name__)


ATTRS = {
    'temperature_2m': dict(
        units='C',
        description="""Near-surface air temperature from ERA5
        (~30km hourly instantaneous)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'temperature_max_2m': dict(
        units='C',
        description="""Near-surface air temperature from ERA5
        (~30km hourly daily max)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'temperature_min_2m': dict(
        units='C',
        description="""Near-surface air temperature from ERA5
        (~30km hourly daily min)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'temperature_mean_2m': dict(
        units='C',
        description="""Near-surface air temperature from ERA5
        (~30km hourly daily mean)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'u_mean_10m': dict(
        units='m/s',
        description="""Near-surface east/west wind from ERA5
        (~30km hourly daily mean)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'v_mean_10m': dict(
        units='m/s',
        description="""Near-surface north/south wind from ERA5
        (~30km hourly daily mean)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'relativehumidity_2m': dict(
        units='%',
        description="""Near-surface relative humidity derived from temperature
        and dew-point from ERA5 (~30km hourly)""",
        valid_min=0,
        valid_max=100,
        dtype='uint16',
        scale_factor=0.01,
    ),
    'sea_surface_temperature': dict(
        units='C',
        description='Sea Surface Temperature from ERA5 (~30km hourly)',
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'lst': dict(
        units='C',
        description="""MODIS land surface temperature gap filled by Iowa State
        and ERA5 sea surface temperature
        (https://doi.org/10.25380/iastate.c.5078492.v3)""",
        valid_min=-100,
        valid_max=100,
        dtype='int16',
        scale_factor=0.01,
    ),
    'evi': dict(
        units='unitless',
        description="""MODIS Enhanced vegitative index MYD13A2 where a greater
        value is more vegitation""",
        valid_min=-0.2,
        valid_max=1,
        dtype='int16',
        scale_factor=0.001,
    ),
    'ghi': dict(
        units='W/m2',
        description='Global Horizontal Irradiance taken from the NSRDB.',
        valid_min=0,
        valid_max=1400,
        dtype='uint16',
        scale_factor=0.1,
    ),
    'dni': dict(
        units='W/m2',
        description='Direct Normal Irradiance taken from the NSRDB.',
        valid_min=0,
        valid_max=1400,
        dtype='uint16',
        scale_factor=0.1,
    ),
    'ghi_mean': dict(
        units='W/m2',
        description="""Daily Average Global Horizontal Irradiance taken from
        the NSRDB.""",
        valid_min=0,
        valid_max=700,
        dtype='uint16',
        scale_factor=0.1,
    ),
    'dni_mean': dict(
        units='W/m2',
        description="""Daily Average Direct Normal Irradiance taken from the
        NSRDB.""",
        valid_min=0,
        valid_max=700,
        dtype='uint16',
        scale_factor=0.1,
    ),
    'albedo_1': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 1 620-670nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_2': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 2 841-876nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_3': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 3 459-479nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_4': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 4 545-565nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_5': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 5 1230-1250nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_6': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 6 1628-1652nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'albedo_7': dict(
        units='unitless',
        description='Surface albedo from MODIS MCD43A4 in band 7 2105-2155nm',
        valid_min=0,
        valid_max=1,
        dtype='uint16',
        scale_factor=0.0001,
    ),
    'topography': dict(
        units='m',
        description='MODIS SRTM topography from SRTMGL3_NC',
        valid_min=-100,
        valid_max=8000,
        dtype='float32',
        scale_factor=1.0,
    ),
    'land_mask': dict(
        units='bool',
        description="""MODIS land mask from land cover type 1 IGBP class from
        MCD12Q1""",
        valid_min=0,
        valid_max=1,
        dtype='int16',
        scale_factor=1.0,
    ),
    'built_volume': dict(
        units='1e3 m3',
        description="""Total built volume per cell derived from the EU global
        human settlement layer""",
        valid_min=0,
        valid_max=1e4,
        dtype='float32',
        scale_factor=1.0,
    ),
    'built_height': dict(
        units='m',
        description="""Average net building height derived from the EU global
        human settlement layer""",
        valid_min=0,
        valid_max=100,
        dtype='float32',
        scale_factor=1.0,
    ),
    'built_surface': dict(
        units='m2',
        description="""Total surface built area derived from the EU global
        human settlement layer""",
        valid_min=0,
        valid_max=1e6,
        dtype='float32',
        scale_factor=1.0,
    ),
    'population': dict(
        units='number of people per cell',
        description="""Population derived from the EU global human settlement
        layer""",
        valid_min=0,
        valid_max=1e5,
        dtype='float32',
        scale_factor=1.0,
    ),
}
for k, v in ATTRS.items():
    ATTRS[k]['standard_name'] = k
    ATTRS[k]['long_name'] = k


HR_OBS = {
    'los_angeles': [10, 21],
    'seattle': [10, 21],
    'houston': [8, 20],
}
"""UTC MODIS observation hours"""


class Utilities:
    """Base utilities for data remapping"""

    @staticmethod
    def make_meta(handle):
        """Make a meta dataframe from the xarray handler

        Parameters
        ----------
        handle : xarray.Dataset
            Open xarray dataset for a MODIS satellite file (e.g.,
            MYD13A2.061_1km_aid0001.nc, MYD11A1.061_1km_aid0001.nc,
            SRTMGL3_NC.003_90m_aid0001.nc). Needs 'lat' and 'lon' variables

        Returns
        -------
        latitude : np.array
            Array of latitude values with shape (latitude, longitude)
        longitude : np.array
            Array of longitude values with shape (latitude, longitude)
        meta : pd.DataFrame
            Meta data with flattened lat/lon and columns "latitude" and
            "longitude"
        """
        longitude = handle['lon'].values
        latitude = handle['lat'].values
        longitude, latitude = np.meshgrid(longitude, latitude)
        meta = {
            'latitude': latitude.flatten(),
            'longitude': longitude.flatten(),
        }
        meta = pd.DataFrame(meta)

        return latitude, longitude, meta

    @staticmethod
    def regrid_data(
        arr, regridder, source_meta, target_meta, target_shape=None
    ):
        """Regrid spatiotemporal data using inverse distance weighted
        interpolation

        Parameters
        ----------
        arr : np.ndarray
            Array of spatio(temporal) data to be regridded. Either a flat 1D
            array of spatial-only data or a 2D array of (space, time). The
            spatial dim must match source_meta.
        regridder : None | sup3r.utilities.Regridder
            None or cached Regridder object. If the target_meta does not match
            the target_meta of the cached object, this object will be
            re-initialized with the new target_meta.
        source_meta : pd.DataFrame
            Source meta data describing input arr, should be standard NREL meta
            data with columns for latitude and longitude
        target_meta : pd.DataFrame
            Target meta data to regrid arr to, should be standard NREL meta
            data with columns for latitude and longitude
        target_shape : tuple | None
            Optional desired final output shape

        Returns
        -------
        arr : np.ndarray
            Same as input but regridded to target_meta and reshaped to
            target_shape (if provided)
        regridder : sup3r.utilities.Regridder
            New Regridder object to cache based on target_meta input
        """

        if target_meta is not None:
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, 1)

            if regridder is not None:
                same_coords = np.allclose(
                    regridder.target_meta.values, target_meta.values
                )

            make_regridder = (
                regridder is None or target_meta is None or not same_coords
            )

            if make_regridder:
                regridder = Regridder(source_meta, target_meta)

            arr = regridder(arr)

            if arr.shape[1] == 1:
                arr = arr[..., 0]

            if target_shape is not None:
                arr = arr.reshape(target_shape)

        return arr, regridder

    @staticmethod
    def nn_map_data(arr, tree, source_meta, target_meta, target_shape=None):
        """
        Parameters
        ----------
        arr : np.ndarray
            Array of spatio(temporal) data to be regridded. Either a flat 1D
            array of spatial-only data or a 2D array of (space, time). The
            spatial dim must match source_meta.
        tree : None | scipy.spatial.KDTree
            None or cached Tree object built on the source meta data.
        source_meta : pd.DataFrame
            Source meta data describing input arr, should be standard NREL meta
            data with columns for latitude and longitude
        target_meta : pd.DataFrame
            Target meta data to regrid arr to, should be standard NREL meta
            data with columns for latitude and longitude
        target_shape : tuple | None
            Optional desired final output shape

        Returns
        -------
        arr : np.ndarray
            Same as input but regridded to target_meta and reshaped to
            target_shape (if provided)
        tree : scipy.spatial.KDTree
            cached Tree object built on the source meta data.
        """

        if target_meta is not None:
            if tree is None:
                tree = KDTree(source_meta.values)

            d, i = tree.query(target_meta.values)

            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, 1)

            regridded = []
            for idt in range(arr.shape[1]):
                iarr = arr[i, idt].reshape((-1, 1))
                regridded.append(iarr)

            arr = np.hstack(regridded)

            if arr.shape[1] == 1:
                arr = arr[..., 0]

            if target_shape is not None:
                arr = arr.reshape(target_shape)

        return arr, tree

    @staticmethod
    def fill_water_data(arr, land_mask, data_range=None):
        """MODIS data has zero vegitation and low albedo over water bodies
        which also have very low surface temperatures. In order to prevent the
        model from learning that black bodies without vegitation are low
        temperature, this method will gap fill the veg/albedo water pixels with
        values that represent the bulk of the land values.

        Parameters
        ----------
        arr : np.ndarray
            3D array (time, lat, lon) typically of evi (vegitation) or albedo
        land_mask : np.ndarray | None
            Optional additional mask that is True where land that can be used
            to perform gap fill of water points without vegitation and low
            albedo. This helps prevent the ML model from learning that no
            vegitation and low albedo results in low surface temperatures. This
            needs to be the same 2D spatial shape as the final output, e.g. the
            same as target_shape
        """

        assert len(arr.shape) == 3
        assert arr.shape[1:] == land_mask.shape

        land_values = np.median(arr, axis=(1, 2))
        arr[:, ~land_mask] = land_values

        return arr

    @staticmethod
    def get_generic_ti(year, h1, h2):
        """Get a generic time index that looks like a midday/midnight modis
        observation series with two observations per day

        Parameters
        ----------
        year : int
            Four digit year
        h1 : int
            First observation hour (nighttime). For LA, this is 10
            (2am local time)
        h2 : int
            Second observation hour (daytime). For LA, this is 21
            (1pm local time)

        Returns
        -------
        ti : pd.DatetimeIndex
            Time index with 2x observations per day
        """

        ti_base = pd.date_range(
            f'{year}0101', f'{year + 1}0101', freq='12h', inclusive='left'
        )
        ti = []
        for idt, timestep in enumerate(ti_base):
            if idt % 2 == 0:
                ti.append(timestep + pd.Timedelta(hours=h1))
            else:
                ti.append(timestep + pd.Timedelta(hours=h2 - 12))
        ti = pd.to_datetime(ti)
        return ti

    @staticmethod
    def get_hr_obs(train_fp):
        """Get observation hours from a previously processed UHI high res
        training data .nc file

        Parameters
        ----------
        train_fp : str
            Filepath to previously processed UHI high res training data .nc
            file

        Returns
        -------
        hours : tuple
            2 entry tuple with (h1, h2) both hours are integers in UTC.
        """
        dset = xr.open_dataset(train_fp)
        ti = pd.to_datetime(dset['time'].values).round(freq='1h')
        counts = ti.groupby(ti.hour)
        counts = {k: len(v) for k, v in counts.items()}
        hours = np.array(list(counts.keys()))
        counts = np.array(list(counts.values()))
        hours = hours[np.argsort(counts)[::-1]][:2]
        h1, h2 = sorted(hours)
        return (h1, h2)

    @staticmethod
    def get_proj_slices(coord, coord_offset, dset):
        """Get x/y slices to subselect a large raster in a non-WGS84 coordinate
        system based on a target coordinate and a +/- offset

        Parameters
        ----------
        coord : tuple
            Coordinate (lat, lon) of the city of interest
        coord_offset : float
            Offset +/- from the coordinate that is being analyzed with
            satellite data. This should be a little bit smaller than the ERA
            raster extent calculated with the pixel nearest the coordinate +/-
            the pixel_offset
        dset : xr.Dataset
            Xarray dataset opening a raster in non-WGS84 coordinate system.
            Must have dset.rio.crs and coordinates x and y

        Returns
        -------
        yslice : slice
            Slice object to select the y dimension of your non-WGS84 raster
            that will include the coord/coord_offset
        xslice : slice
            Slice object to select the x dimension of your non-WGS84 raster
            that will include the coord/coord_offset
        """

        assert 'x' in dset.coords
        assert 'y' in dset.coords
        assert dset.rio.crs is not None

        n = 10
        y_target = np.linspace(
            coord[0] + coord_offset, coord[0] - coord_offset, n
        )
        x_target = np.linspace(
            coord[1] - coord_offset, coord[1] + coord_offset, n
        )
        bounds = xr.Dataset(
            {
                'var': (('y', 'x'), np.ones((n, n))),
            },
            coords={
                'y': (('y',), y_target),
                'x': (('x',), x_target),
            },
        )
        bounds = bounds.rio.write_crs('EPSG:4326')  # set WGS84
        bounds = bounds.rio.write_transform()
        bounds = bounds.rio.reproject(dset.rio.crs)

        xslice = (dset.x > bounds.x.min()) & (dset.x < bounds.x.max())
        yslice = (dset.y > bounds.y.min()) & (dset.y < bounds.y.max())
        xslice = np.where(xslice)[0]
        yslice = np.where(yslice)[0]
        if len(xslice) <= 1 or len(yslice) <= 1:
            raise RuntimeError(
                'Could not find any data close to coord '
                f'{coord} with offset {coord_offset}'
            )
        xslice = slice(xslice[0], xslice[-1] + 1)
        yslice = slice(yslice[0], yslice[-1] + 1)

        return yslice, xslice


class ModisRawLstProduct:
    """Class to handle raw MODIS LST data products (MYD11A1.061), convert
    MODIS view time to UTC conversion, and convert quality flags to cloud mask
    """

    def __init__(self, fp, daynight):
        """
        Parameters
        ----------
        fp : str
            Downloaded MODIS satellite data from the appeears API.
            File type : "netcdf4" with extension ".nc"
            Projection : "geographic"
            Datasets : Day_view_time, LST_Day_1km, QC_Day, etc...
            MODIS product ID : MYD11A1.061
            Spatial extent : Flexible but usually one coord +/- 1 dec. deg.
            Spatial resolution : 1km
            Temporal extent : Flexible but usually one year
            Temporal resolution : Daily (for a night or day file)
            Example filename : MYD11A1.061_1km_aid0001.nc
        daynight : str
            Either "day" or "night" depending on modis observation time
        """

        self.handle = xr.open_dataset(fp)
        self.daynight = daynight.lower().title()
        self._time_index = None
        self.regrid = None
        meta = Utilities.make_meta(self.handle)
        self.latitude, self.longitude, self.meta = meta
        self.shape = self.latitude.shape

    @property
    def time_index(self):
        """Get the MODIS file time index in UTC. This converts the MODIS
        viewing time which is in local solar time to UTC

        day/night view time = UTC + (longitude / 15)
        +24 if local solar time < 0
        -24 if local solar time >=24

        https://lpdaac.usgs.gov/documents/118/MOD11_User_Guide_V6.pdf

        Returns
        -------
        pd.DatetimeIndex
        """

        if self._time_index is None:
            time_index = [ts.isoformat() for ts in self.handle['time'].values]
            time_index = pd.to_datetime(time_index)

            dset = self.handle[f'{self.daynight}_view_time']

            # always raises "Mean of empty slice" warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                view_times = np.nanmean(dset.values, axis=(1, 2))

            view_times = np.nan_to_num(view_times, nan=np.nanmean(view_times))
            view_times = view_times - self.handle['lon'].values.mean() / 15
            view_times[view_times < 0] += 24
            view_times[view_times >= 24] -= 24

            hours = [int(vt) for vt in view_times]
            minutes = [int((vt % 1) * 60) for vt in view_times]

            time_index = [
                mti + timedelta(hours=hours[i], minutes=minutes[i])
                for i, mti in enumerate(time_index)
            ]

            self._time_index = pd.to_datetime(time_index)

        return self._time_index

    def get_data(
        self,
        target_meta,
        target_shape,
        s_enhance=None,
        dset='LST_{daynight}_1km',
    ):
        """Get datasets from the raw MODIS data

        Parameters
        ----------
        target_meta : pd.DataFrame
            Target meta data to interpolate the 1km raw data to, should be
            standard NREL meta data with columns for latitude and longitude
        target_shape : tuple
            2-entry tuple with desired (latitude, longitude) output shape
        dset : str
            Optional LST dataset name to retrieve from the raw MODIS file. If
            there is a {daynight} format variable available, it will be filled
            based on this class's init argument

        Returns
        -------
        arr : np.array
            Raw MODIS LST data in shape (time, latitude, longitude)
        """

        if '{daynight}' in dset:
            dset = dset.format(daynight=self.daynight)

        # extract and reshape to (lat, lon, time)
        arr = self.handle[dset][...].values
        arr = np.transpose(arr, (1, 2, 0))

        arr = arr.reshape((arr.shape[0] * arr.shape[1], -1))
        arr, self.regrid = Utilities.regrid_data(
            arr, self.regrid, self.meta, target_meta
        )

        arr = arr.reshape(target_shape + (-1,))
        arr = np.transpose(arr, (2, 0, 1))

        return arr


class ModisStaticLayer:
    """Class to handle MODIS static layers like elevation (SRTM) or land cover
    data"""

    def __init__(self, fp, dset='SRTMGL3_DEM'):
        """
        Parameters
        ----------
        fp : str
            Downloaded MODIS satellite data from the appeears API.
            File type : "netcdf4" with extension ".nc"
            Projection : "geographic"
            Datasets : SRTMGL3_DEM (see dset input) or LC_Type1
            MODIS product ID : SRTMGL3_NC.003 or MCD12Q1.061
            Spatial extent : Flexible but usually one coord +/- 1 dec. deg.
            Spatial resolution : higher res than target meta, e.g. 90m or 500m
            Temporal extent : Single timestep
            Temporal resolution : Single timestep
            Example filename : SRTMGL3_NC.003_90m_aid0001.nc or
                MCD12Q1.061_500m_aid0001.nc
        dset : str
            Dataset to be loaded, typically SRTMGL3_DEM or LC_Type1
        """
        self.fp = fp
        self.handle = xr.open_dataset(fp)
        self.dset = dset
        assert self.dset in self.handle

        meta = Utilities.make_meta(self.handle)
        self.latitude, self.longitude, self.meta = meta
        self.lat_lon = np.dstack((self.latitude, self.longitude))

    def get_data(self, target_meta, target_shape):
        """Get static layer data from the modis file aggregated and mapped to a
        target meta data

        Parameters
        ----------
        target_meta : pd.DataFrame
            Target meta data to aggregate and map the high-res static layer to,
            the static layer should be higher-resolution than this input,
            should be standard NREL meta data with columns for latitude and
            longitude
        target_shape : tuple
            2-entry tuple with shape in order of (lat, lon)

        Returns
        -------
        arr : np.array
            2D output array with static layer values reshaped to target_shape
        """

        arr = self.handle[self.dset].values
        if 'time' in self.handle.coords and arr.shape[0] == 1:
            arr = arr[0]

        missing = np.isnan(arr)
        if missing.any():
            arr = nn_fill_array(arr)

        tree = KDTree(target_meta[['latitude', 'longitude']].values)
        d, i = tree.query(self.meta[['latitude', 'longitude']].values)

        df = pd.DataFrame({'data': arr.flatten(), 'gid_target': i, 'd': d})
        df = df[df['gid_target'] != len(target_meta)]
        df = df.sort_values('gid_target')
        df = df.groupby('gid_target').mean()

        missing = set(np.arange(len(target_meta))) - set(df.index)
        if len(missing) > 0:
            temp_df = pd.DataFrame({'topo': np.nan}, index=sorted(missing))
            df = pd.concat((df, temp_df)).sort_index()

        arr = df['data'].values.reshape(target_shape)
        arr = nn_fill_array(arr)

        return arr


class ModisVeg:
    """Class to handle MODIS vegitative indices data (e.g., EVI)"""

    def __init__(self, fp, dset='_1_km_16_days_EVI'):
        """
        Parameters
        ----------
        fp : str
            Downloaded MODIS satellite data from the appeears API.
            File type : "netcdf4" with extension ".nc"
            Projection : "geographic"
            Datasets : _1_km_16_days_EVI (see dset input)
            MODIS product ID : MYD13A2.061
            Spatial extent : Flexible but usually one coord +/- 1 dec. deg.
            Spatial resolution : 1km
            Temporal extent : Flexible but usually one year
            Temporal resolution : Every 16 days
            Example filename : MYD13A2.061_1km_aid0001.nc
        dset : str
            Dataset to be loaded, typically _1_km_16_days_EVI or
            _1_km_16_days_NDVI but EVI appears to be the more modern output
        """

        self.handle = xr.open_dataset(fp)
        self.dset = dset
        assert self.dset in self.handle

        meta = Utilities.make_meta(self.handle)
        self.latitude, self.longitude, self.meta = meta
        self.lat_lon = np.dstack((self.latitude, self.longitude))

        self.time_index = [ts.isoformat() for ts in self.handle['time'].values]
        self.time_index = pd.to_datetime(self.time_index)

        self.regrid = None

    def get_t_interp(self, target_time_index):
        """Get the MODIS vegitation data interpolated to target temporal
        index

        Parameters
        ----------
        target_time_index : pd.DatetimeIndex
            Target time index to interpolate the raw 16-day veg data to.

        Returns
        -------
        evi : np.array
            Enhanced vegitative index with shape (time, latitude, longitude)
            data is from (-0.2 to 1) where 1 is more vegitation.
        """

        arr = self.handle[self.dset].values
        arr = np.nan_to_num(arr, nan=-0.2).astype(np.float32)

        full_time_shape = (len(target_time_index), arr.shape[1], arr.shape[2])
        arr_interp = np.zeros(full_time_shape, dtype=np.float32)

        for idt, ts in enumerate(target_time_index):
            dt = np.abs(ts - self.time_index)
            idt0, idt1 = sorted(np.argsort(dt)[:2])
            x0, x1 = dt[idt0].seconds, dt[idt1].seconds

            iarr = (arr[idt0] * x0 + arr[idt1] * x1) / (x0 + x1)
            arr_interp[idt] = iarr

        return arr_interp

    def get_data(
        self, target_time_index, target_meta, target_shape, land_mask=None
    ):
        """Get the MODIS vegitation data interpolated to target spatiotemporal
        meta data

        Parameters
        ----------
        target_time_index : pd.DatetimeIndex
            Target time index to interpolate the raw 16-day veg data to.
        target_meta : pd.DataFrame
            Target meta data to interpolate the 1km veg data to, should be
            standard NREL meta data with columns for latitude and longitude
        target_shape : tuple
            2-entry tuple with desired (latitude, longitude) output shape
        land_mask : np.ndarray | None
            Optional additional mask that is True where land that can be used
            to perform gap fill of water points without vegitation and low
            albedo. This helps prevent the ML model from learning that no
            vegitation and low albedo results in low surface temperatures. This
            needs to be the same 2D spatial shape as the final output, e.g. the
            same as target_shape

        Returns
        -------
        evi : np.array
            Enhanced vegitative index with shape (time, latitude, longitude)
            data is from (-0.2 to 1) where 1 is more vegitation.
        """

        evi = self.get_t_interp(target_time_index)
        evi = np.transpose(evi, (1, 2, 0))
        evi = evi.reshape((evi.shape[0] * evi.shape[1], -1))

        evi, self.regrid = Utilities.regrid_data(
            evi, self.regrid, self.meta, target_meta
        )

        evi = evi.reshape(target_shape + (-1,))
        evi = np.transpose(evi, (2, 0, 1))

        if land_mask is not None:
            evi = Utilities.fill_water_data(
                evi, land_mask, data_range=(-0.2, 1)
            )

        return evi


class ModisAlbedo:
    """Class to handle MODIS albedo data"""

    def __init__(self, fp):
        """
        Parameters
        ----------
        fp : str
            Downloaded MODIS satellite data from the appeears API.
            File type : "netcdf4" with extension ".nc"
            Projection : "geographic"
            Datasets : Albedo_WSA_shortwave, Albedo_WSA_nir, Albedo_WSA_vis
            MODIS product ID : MCD43A3.061 | MCD43A4.061
            Spatial extent : Flexible but usually one coord +/- 1 dec. deg.
            Spatial resolution : 500m
            Temporal extent : Flexible but usually one year
            Temporal resolution : Every day with gaps (NaNs)
            Example filename : MCD43A3.061_500m_aid0001.nc or
                MCD43A4.061_500m_aid0001.nc
        """

        self.handle = xr.open_dataset(fp)

        meta = Utilities.make_meta(self.handle)
        self.latitude, self.longitude, self.meta = meta
        self.lat_lon = np.dstack((self.latitude, self.longitude))
        self.shape = self.latitude.shape

        self.time_index = [ts.isoformat() for ts in self.handle['time'].values]
        self.time_index = pd.to_datetime(self.time_index)
        self.year = self.time_index.year[0]
        self.full_time_index = pd.date_range(
            f'{self.year}0101',
            f'{self.year + 1}0101',
            freq='1d',
            inclusive='left',
        )

        self.regrid = None

    def get_data(
        self,
        dset,
        dset_qa=None,
        coarsen=None,
        land_mask=None,
        t_interp=True,
        s_interp=True,
    ):
        """Get MODIS albedo data coarsened from 500m native resolution to the
        1km MODIS LST resolution

        Parameters
        ----------
        dset : str
            Dataset to be loaded, typically Nadir_Reflectance_Band{i} for
            MCD43A4 or Albedo_WSA_shortwave for MCD43A3
        dset_qa : str
            Optional QA dataset for the MODIS albedo data, e.g.,
            BRDF_Albedo_Band_Mandatory_Quality_Band1 for MCD43A4 where 0 is
            good quality albedo data. If provided, albedo data where
            ``quality != 0`` will be set to NaN and gap filled. This is not
            always advisable because (for example) may pixels in downtown LA
            never get high-quality albedo retrieval, so they would be filled
            using spatial nearest neighbor. In this case, it is possible that a
            low-quality albedo retrieval is preferable to inappropriate spatial
            gap fill.
        coarsen : None | int
            Optional factor by which to coarsen the spatial axes.
        land_mask : np.ndarray | None
            Optional additional mask that is True where land. Albedo will be 0
            for water.
        t_interp : bool
            Option to interpolate missing albedo data for pixels across time
        s_interp : bool
            Option to interpolate missing albedo data for pixels from their
            nearest neighbors (happens after t_interp)

        Returns
        -------
        arr : np.ndarray
            3D surface albedo data with shape (time, lat, lon) where time is
            per day.
        """

        assert dset in self.handle

        arr = self.handle[dset].values
        shape_3d = list(arr.shape)
        shape_2d = (arr.shape[0], arr.shape[1] * arr.shape[2])

        if dset_qa is not None:
            arr_qa = self.handle[dset_qa].values
            nan_mask = arr_qa != 0  # 0 is good quality
            arr[nan_mask] = np.nan

        # use pandas to interpolate across time by site
        if t_interp:
            df = pd.DataFrame(arr.reshape(shape_2d), index=self.time_index)
            df = df.reindex(self.full_time_index)
            df = df.interpolate('linear').ffill().bfill()
            shape_3d[0] = len(df)
            arr = df.values

        # reshape to (time, lat, lon)
        arr = arr.reshape(shape_3d)

        # interpolate the rest of the data
        if s_interp:
            arr = [nn_fill_array(arr[i]) for i in range(len(arr))]
            arr = np.dstack(arr)
            arr = np.transpose(arr, (2, 0, 1))

        # coarsen 500m albedo to 1km lst grid
        if coarsen is not None:
            arr = spatial_coarsening(arr, s_enhance=coarsen, obs_axis=True)

        if land_mask is not None:
            arr[:, ~land_mask] = 0

        return arr


class Nsrdb:
    """Class to handle NSRDB .h5 data and munge into Sup3rUHI training data .nc
    format"""

    def __init__(self, fp, coord, coord_offset):
        """
        Parameters
        ----------
        fp : str
            Filepath to .h5 NSRDB file.
        coord : tuple
            Coordinate (lat, lon) of the city of interest
        coord_offset : float
            Offset +/- from the coordinate that is being analyzed with
            satellite data. This should be a little bit smaller than the ERA
            raster extent calculated with the pixel nearest the coordinate +/-
            the pixel_offset
        """

        Handle = Resource
        if '*' in fp or isinstance(fp, (list, tuple)):
            Handle = MultiFileResource

        self.handle = Handle(fp)
        self.meta = self.handle.meta

        lat_mask = self.meta['latitude'].values > (coord[0] - coord_offset)
        lat_mask &= self.meta['latitude'].values < (coord[0] + coord_offset)
        lon_mask = self.meta['longitude'].values > (coord[1] - coord_offset)
        lon_mask &= self.meta['longitude'].values < (coord[1] + coord_offset)

        self.mask = lon_mask & lat_mask
        self.iloc = np.where(self.mask)[0]
        self.regrid = None

    def _daily_reduce(self, arr, target_time_index, daily_reduce=None):
        """Reduce an instantaneous NSRDB (time, space) 2D array to daily values

        Parameters
        ----------
        arr : np.ndarray
            NSRDB data array in shape (time, space) where time matches
            self.handle.time_index
        target_time_index : pd.DatetimeIndex
            Target time index to map NSRDB data to

        Returns
        -------
        out : np.ndarray
            Reduced array with daily values in 0 axis.
        """

        if daily_reduce is None:
            return arr

        tslices = []
        source_time_index = self.handle.time_index.tz_localize(None)
        for tstamp in target_time_index:
            mask = tstamp.date() == source_time_index.date
            ilocs = np.where(mask)[0]
            tslices.append(slice(ilocs[0], ilocs[-1] + 1))

        if daily_reduce.casefold() == 'max':
            out = np.vstack([arr[tslice].max(0) for tslice in tslices])
        elif daily_reduce.casefold() == 'min':
            out = np.vstack([arr[tslice].min(0) for tslice in tslices])
        elif daily_reduce.casefold() == 'mean':
            out = np.vstack([arr[tslice].mean(0) for tslice in tslices])

        return out

    def _get_time_indices(self, target_time_index):
        """Get indices to slice the NSRDB time index that will yield the
        target_time_index

        Parameters
        ----------
        target_time_index : jpd.DatetimeIndex
            Target time index to map NSRDB data to

        Returns
        -------
        idts : list
            List of indices that when used to slice the NSRDB time index will
            yield output aligning with the target_time_index
        """

        source_time_index = self.handle.time_index.tz_localize(None)
        idts = []
        for time_stamp in target_time_index:
            diff = source_time_index - time_stamp
            idt = np.argmin(np.abs(diff))
            idts.append(idt)
        return idts

    def get_data(
        self,
        dset,
        target_time_index,
        target_meta,
        target_shape,
        daily_reduce=None,
    ):
        """Get timeseries data from NSRDB mapped to target meta data

        Parameters
        ----------
        dset : str
            Dataset name from the target NSRDB file (e.g., "dni")
        target_time_index : pd.DatetimeIndex
            Target time index to map NSRDB data to
        target_meta : pd.DataFrame
            Target meta data to interpolate the 1km raw data to, should be
            standard NREL meta data with columns for latitude and longitude
        target_shape : tuple
            2-entry tuple with desired (latitude, longitude) output shape
        daily_reduce : str | None
            Option to do a daily reduction of source data. This can be "min",
            "max", "mean" or None for instantaneous (default)

        Returns
        -------
        arr : np.array
            NSRDB data reshaped to (time, latitude, longitude)
        """
        arr = self.handle[dset, :, self.iloc]

        if daily_reduce is None:
            idts = self._get_time_indices(target_time_index)
            arr = arr[idts, :]
        else:
            arr = self._daily_reduce(arr, target_time_index, daily_reduce)

        arr = np.transpose(arr, axes=(1, 0))
        meta = self.meta.iloc[self.iloc].reset_index(drop=True)
        arr, self.regrid = Utilities.regrid_data(
            arr, self.regrid, meta, target_meta
        )
        arr = arr.reshape(target_shape + (-1,))
        arr = np.transpose(arr, axes=(2, 0, 1))
        return arr


class ModisGfLst:
    """Class to handle and remap MODIS gap-filled LST data from the following
    reference. Note that this uses the raw sinusoidal projection .tiff data

    Zhang, T., Zhou, Y., Zhu, Z., Li, X., and Asrar, G. R.: A global seamless
    1 km resolution daily land surface temperature dataset (2003–2020), Earth
    Syst. Sci. Data, 14, 651–664, https://doi.org/10.5194/essd-14-651-2022,
    2022.

    Data:
    (https://iastate.figshare.com/collections/A_global_seamless_1_km_
     resolution_daily_land_surface_temperature_dataset_2003_2020_/
     5078492)
    """

    def __init__(self, fp, coord, coord_offset, yslice=None, xslice=None):
        """
        Parameters
        ----------
        fp : str
            Filepath to .tiff that is the MODIS gap-filled LST file from
            IA State with sinusoidal x/y coordinates. LST is
            in 'band_data' dataset and is in units of 0.1C. Needs datasets:
            'band_data' which has shape (1, y, x), y, and x. Note that this
            data drops 12/31 in leap years and has missing values offshore.
        coord : tuple
            Coordinate (lat, lon) of the city of interest
        coord_offset : float
            Offset +/- from the coordinate that is being analyzed with
            satellite data. This should be a little bit smaller than the ERA
            raster extent calculated with the pixel nearest the coordinate +/-
            the pixel_offset
        yslice : slice
            Slice object to select the y dimension of the non-WGS84 LST raster
            that will include the coord/coord_offset
        xslice : slice
            Slice object to select the x dimension of the non-WGS84 LST raster
            that will include the coord/coord_offset
        """

        assert fp.endswith(('.tiff', '.tif'))
        self.handle = xr.open_dataset(fp)
        assert 'band_data' in self.handle
        assert 'x' in self.handle.coords
        assert 'y' in self.handle.coords

        if yslice is not None and xslice is not None:
            self.yslice, self.xslice = yslice, xslice
        else:
            self.yslice, self.xslice = Utilities.get_proj_slices(
                coord, coord_offset, self.handle
            )

        self.handle = self.handle.isel(y=self.yslice, x=self.xslice)
        self.handle = self.handle.rio.reproject('EPSG:4326')

        self.latitude = self.handle['y']
        self.longitude = self.handle['x']
        self.longitude, self.latitude = np.meshgrid(
            self.longitude, self.latitude
        )
        self.lat_lon = np.dstack((self.latitude, self.longitude))
        self.shape = self.longitude.shape

        self.meta = {
            'latitude': self.latitude.flatten(),
            'longitude': self.longitude.flatten(),
        }
        self.meta = pd.DataFrame(self.meta)

        self.regrid = None

    def get_data(
        self,
        era_temp=None,
        target_meta=None,
        target_shape=None,
        land_mask=None,
    ):
        """Get MODIS LST data for the requested coordinate +/- offset

        Parameters
        ----------
        era_temp : np.array | None
            ERA temperature (preferably sea_surface_temperature) data to use to
            fill any missing data in the source MODIS LST fields typically
            caused by water pixels. If None, NaNs will be present in the
            output. This needs to be the same shape as the final output, e.g.
            the same as target_shape
        target_meta : pd.DataFrame
            Target meta data to regrid the IAState LST data to, should be
            standard NREL meta data with columns for latitude and longitude
        target_shape : tuple
            2-entry tuple with shape in order of (lat, lon)
        land_mask : np.ndarray | None
            Optional additional mask that is True where land that can be used
            to perform an additional nearest-neighbor fill of land points that
            are erroneously NaN in the IAState data. This needs to be the same
            2D spatial shape as the final output, e.g. the same as target_shape

        Returns
        -------
        lst : np.ndarray
            Output MODIS land surface temperature (LST) in degrees Celsius for
            a single observation in 2D shape (lat, lon). Data is originally
            land gap-filled LST from IA State university with ERA t2m data
            added in for offshore pixels. This is on the native MODIS grid
            (sinusoidal) if target_meta is None and is regridded to target_meta
            if not.
        """

        lst = self.handle['band_data'][0, :, :].values

        nan_mask = np.isnan(lst)
        lst = lst.astype(np.float32)
        lst /= 10  # Gap-filled MODIS LST data scaled to 0.1C

        if target_meta is not None:
            lst, self.regrid = Utilities.regrid_data(
                lst.flatten(),
                self.regrid,
                self.meta,
                target_meta,
                target_shape,
            )

        if land_mask is not None:
            lst = nn_fill_array(lst)
            lst[~land_mask] = np.nan

        if era_temp is not None:
            if era_temp.shape != lst.shape:
                try:
                    era_temp = era_temp.reshape(lst.shape)
                except Exception as e:
                    msg = (
                        'Could not reshape era_temp input with shape '
                        '{} to LST shape of {}. The era_temp must be '
                        'regridded to ModisGfLst.meta!'.format(
                            era_temp.shape, lst.shape
                        )
                    )
                    raise RuntimeError(msg) from e
            nan_mask = np.isnan(lst)
            lst[nan_mask] = era_temp[nan_mask]

        return lst


class GhsData:
    """Class to handle 100m Global Human Settlement (GHS) data from:
    https://ghsl.jrc.ec.europa.eu/download.php?
    """

    def __init__(self, fps, coord, coord_offset, yslice=None, xslice=None):
        """
        Parameters
        ----------
        fps : str | list
            One or more filepaths to GHS data files in .tif with EU GHSL
            projection (i think World_Mollweide CRS). Needs: CRS information,
            band_data, x, and y.
        coord : tuple
            Coordinate (lat, lon) of the city of interest
        coord_offset : float
            Offset +/- from the coordinate that is being analyzed with
            satellite data. This should be a little bit smaller than the ERA
            raster extent calculated with the pixel nearest the coordinate +/-
            the pixel_offset
        yslice : slice
            Slice object to select the y dimension of the non-WGS84 LST raster
            that will include the coord/coord_offset
        xslice : slice
            Slice object to select the x dimension of the non-WGS84 LST raster
            that will include the coord/coord_offset
        """

        if isinstance(fps, str):
            fps = sorted(glob.glob(fps))
        assert isinstance(fps, (list, tuple))

        self.fps = []
        self.handles = []
        for fp in fps:
            handle = xr.open_dataset(fp)
            assert 'band_data' in handle
            assert 'x' in handle.coords
            assert 'y' in handle.coords
            try:
                yslice, xslice = Utilities.get_proj_slices(
                    coord, coord_offset, handle
                )
            except RuntimeError as _:
                logger.debug(f'GHSL w/ no nearby pixels, ignoring: {fp}')
            else:
                logger.debug(f'GHSL good extent: {fp}')
                handle = handle.isel(y=yslice, x=xslice)
                handle = handle.rio.reproject('EPSG:4326')
                self.handles.append(handle)
                self.fps.append(fp)

        if len(self.handles) > 1 or len(self.handles) == 0:
            msg = (
                f'Found {len(self.handles)} GHSL files with valid data, '
                'multi file concatenation needs more testing'
            )
            raise NotImplementedError(msg)

        self.latitude = []
        self.longitude = []
        for handle in self.handles:
            latitude = handle['y']
            longitude = handle['x']
            longitude, latitude = np.meshgrid(longitude, latitude)
            self.longitude.append(longitude.flatten())
            self.latitude.append(latitude.flatten())

        self.longitude = np.concatenate(self.longitude, axis=0)
        self.latitude = np.concatenate(self.latitude, axis=0)

        self.meta = {'latitude': self.latitude, 'longitude': self.longitude}
        self.meta = pd.DataFrame(self.meta)

        self.regrid = None

    def get_data(self, target_meta, target_shape, mode='mean'):
        """Get the GHS data mapped to a target meta / shape.

        This aggregates 100m GHS data to lower resolution meta data by subgrid
        nearest neighbor

        Parameters
        ----------
        target_meta : pd.DataFrame
            Target meta data to aggregate and map the 100m GHS data to,
            should be standard NREL meta data with columns for latitude and
            longitude
        target_shape : tuple
            2-entry tuple with shape in order of (lat, lon)
        mode : str
            Aggregation mode (mean, sum, min, max)

        Returns
        -------
        out : np.ndarray
            Raster of GHS data mapped to target_meta and reshaped to
            target_shape. Final shape should be (lat, lon) with no
            time-dependence.
        """

        arr = []
        for handle in self.handles:
            arr.append(handle['band_data'][0, :, :].values.flatten())
        arr = np.concatenate(arr, axis=0)

        tree = KDTree(target_meta[['latitude', 'longitude']].values)
        d, i = tree.query(self.meta[['latitude', 'longitude']].values)

        df = pd.DataFrame({'data': arr, 'gid_target': i, 'd': d})
        df = df[df['gid_target'] != len(target_meta)]
        df = df.sort_values('gid_target')

        if mode.casefold() == 'mean':
            df = df.groupby('gid_target').mean()
        elif mode.casefold() == 'sum':
            df = df.groupby('gid_target').sum()
        elif mode.casefold() == 'min':
            df = df.groupby('gid_target').min()
        elif mode.casefold() == 'max':
            df = df.groupby('gid_target').max()
        else:
            raise ValueError(f'Bad mode: "{mode}"')

        missing = set(np.arange(len(target_meta))) - set(df.index)
        if len(missing) > 0:
            temp_df = pd.DataFrame({'topo': np.nan}, index=sorted(missing))
            df = pd.concat((df, temp_df)).sort_index()

        arr = df['data'].values.reshape(target_shape)
        arr = nn_fill_array(arr)

        return arr


class EraCity:
    """Class to handle data from an ERA file for a small raster extent around a
    single city"""

    def __init__(self, fp, coord, coord_offset, pixel_offset, s_enhance):
        """
        Parameters
        ----------
        fp : str
            ERA .nc filepath with one year of ~31km hourly data. Dataset shape
            is expected to be (time, lat, lon). Valid 3D surface datasets:
            temperature_2m, relativehumidity_2m, sea_surface_temperature,
            u_10m, v_10m
        coord : tuple
            Coordinate (lat, lon) of the city of interest
        coord_offset : float
            Offset +/- from the coordinate that is being analyzed with
            satellite data. This should be a little bit smaller than the ERA
            raster extent calculated with the pixel nearest the coordinate +/-
            the pixel_offset
        pixel_offset : int
            Number of ERA pixels +/- the pixel nearest to coord. This
            determines the ERA raster being processed and should be slightly
            bigger than the satellite extent being examined with coord +/-
            coord_offset
        s_enhance : int
            Spatial enhancement multiplier that the ERA data is being enhanced
            by. For example, when using ~30km ERA5 and ~1km MODIS s_enhance is
            typically 30
        """

        self.handle = xr.open_dataset(fp)
        self.s_enhance = s_enhance

        lon, lat = self.handle['lon'].values, self.handle['lat'].values
        lon, lat = np.meshgrid(lon, lat)
        diff_lat = lat - coord[0]
        diff_lon = lon - coord[1]
        diff = np.hypot(diff_lat, diff_lon)
        idy, idx = np.where(np.min(diff) == diff)
        idy, idx = idy[0], idx[0]

        self.yslice = slice(idy - pixel_offset, idy + pixel_offset + 1)
        self.xslice = slice(idx - pixel_offset, idx + pixel_offset + 1)
        lat = lat[self.yslice, self.xslice]
        lon = lon[self.yslice, self.xslice]
        self.lr_lat_lon = np.dstack((lat, lon))

        self.hr_shape = (
            self.lr_lat_lon.shape[0] * self.s_enhance,
            self.lr_lat_lon.shape[1] * self.s_enhance,
        )

        self.hr_lat_lon = OutputHandler.get_lat_lon(
            self.lr_lat_lon, self.hr_shape
        )

        self.hr_meta = {
            'latitude': self.hr_lat_lon[..., 0].flatten(),
            'longitude': self.hr_lat_lon[..., 1].flatten(),
        }
        self.hr_meta = pd.DataFrame(self.hr_meta)

        self.lr_meta = {
            'latitude': self.lr_lat_lon[..., 0].flatten(),
            'longitude': self.lr_lat_lon[..., 1].flatten(),
        }
        self.lr_meta = pd.DataFrame(self.lr_meta)

        self.regrid = None

    def get_lr_dset(self, dset, timestamp, daily_reduce):
        """Get low-resolution data from ERA5 file."""
        assert self.handle[dset].shape[0] == self.handle['time'].shape[0]
        assert self.handle[dset].shape[1] == self.handle['lat'].shape[0]
        assert self.handle[dset].shape[2] == self.handle['lon'].shape[0]

        era_ti = pd.to_datetime(self.handle['time'].values)

        if daily_reduce is None:
            idt0, idt1 = sorted(np.argsort(np.abs(era_ti - timestamp))[:2])
            x1_x0 = (era_ti[idt1] - era_ti[idt0]).seconds
            x_x0 = (timestamp - era_ti[idt0]).seconds
            y0 = self.handle[dset][idt0, self.yslice, self.xslice].values
            y1 = self.handle[dset][idt1, self.yslice, self.xslice].values
            out = y0 + x_x0 * (y1 - y0) / (x1_x0)
        else:
            idt0 = np.where(era_ti.date == timestamp.date())[0]
            out = self.handle[dset][idt0, self.yslice, self.xslice].values
            if daily_reduce.casefold() == 'mean':
                out = out.mean(0)
            elif daily_reduce.casefold() == 'max':
                out = out.max(0)
            elif daily_reduce.casefold() == 'min':
                out = out.min(0)

        out = np.expand_dims(out, 0)
        return out

    def _parallel_interp(self, arr, model, exo_data, max_workers):
        """Run SurfaceSpatialMetModel interpolation in parallel on chunks of
        data.

        Parameters
        ----------
        arr : np.ndarray
            4D array of ERA data in shape (time, lat, lon, features)
        model : SurfaceSpatialMetModel
            Initialized model ready to perform interpolation
        exo_data : dict
            Exogeneous data including high-res topography that can be passed to
            SurfaceSpatialMetModel.generate
        max_workers : int | None
            Number of parallel works to use to perform interpolation of ERA
            temp/humidity data. 1 is serial, None is all available.

        Returns
        -------
        arr : np.ndarray
            Same as input arr but with enhanced lat, lon dimensions
        """
        output = []
        futures = []
        n_chunks = max_workers or os.cpu_count()
        chunks = np.array_split(arr, n_chunks, axis=0)
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            for chunk in chunks:
                future = exe.submit(
                    model.generate, chunk, exogenous_data=exo_data
                )
                futures.append(future)

            for future in futures:
                output.append(future.result())

        out = np.vstack(output)
        return out

    def get_data(
        self,
        dsets,
        time,
        hr_topo=None,
        interpolate=True,
        daily_reduce=None,
        target_meta=None,
        target_shape=None,
        max_workers=1,
    ):
        """This method takes the ~31km ERA data, uses interpolation methods
        from Sup3r and then regrids the high-res data onto the target meta
        data. The sup3r interp methods includes: lanczos interpolation, t/rh
        lapse rates, low-resolution bias adjustment vs. original inputs

        Parameters
        ----------
        dsets : str | list
            Dataset name to get from the ERA file.
        time : pd.Timestamp | slice
            Time slice or Timestamp to interpolate ERA data to. Timestamp is
            typically one entry from ModisRawLstProduct.time_index
        hr_topo : np.ndarray
            2D array of high-resolution topography with (lat, lon) shape
            equivelant to the ERA city shape after s_enhance used for
            interpolate with lapse rate. Note that this cant just be the
            MODIS/SRTM topography but needs to be mapped/aggregated to the
            enhanced ERA grid.
        interpolate : bool
            Flag to interpolate to higher resolution based on s_enhance
        daily_reduce : str | None
            Option to do a daily reduction of source data. This can be "min",
            "max", "mean" or None for instantaneous (default)
        target_meta : pd.DataFrame
            Target meta data to aggregate and map the interpolated ERA data to,
            should be standard NREL meta data with columns for latitude and
            longitude
        target_shape : tuple
            2-entry tuple with shape in order of (lat, lon)
        max_workers : int | None
            Number of parallel works to use to perform interpolation of ERA
            temp/humidity data. 1 is serial, None is all available.

        Returns
        -------
        out : np.ndarray
            3D array of ERA data with shape (lat, lon, features) if time is a
            single timestamp to interpoalte to, or 4D array
            (time, lat, lon, features) if time is a slice
        """

        dsets = dsets if isinstance(dsets, (list, tuple)) else [dsets]
        arr = []
        for dset in dsets:
            if isinstance(time, pd.Timestamp):
                iarr = self.get_lr_dset(dset, time, daily_reduce)
            elif isinstance(time, slice):
                iarr = self.handle[dset][time, self.yslice, self.xslice]
                iarr = iarr.values
            elif time is None:
                iarr = self.handle[dset][:, self.yslice, self.xslice].values
            else:
                raise ValueError(f'Bad time input: {time}')

            iarr = np.expand_dims(iarr, -1)
            arr.append(iarr)

        arr = np.concatenate(arr, -1)

        if interpolate:
            assert hr_topo is not None
            hr_topo = nn_fill_array(hr_topo)
            model = SurfaceSpatialMetModel(dsets, self.s_enhance)
            lr_topo = spatial_coarsening(
                np.expand_dims(hr_topo, -1),
                s_enhance=self.s_enhance,
                obs_axis=False,
            )[..., 0]
            exo_data = [{'data': lr_topo}, {'data': hr_topo}]
            exo_data = {'topography': {'steps': exo_data}}

            # SurfaceSpatialMetModel requires 4D (obs, space, space, features)
            if max_workers == 1:
                arr = model.generate(arr, exogenous_data=exo_data)
            else:
                arr = self._parallel_interp(arr, model, exo_data, max_workers)

        if target_meta is not None:
            regridded_arr = []
            for idf in range(arr.shape[-1]):
                iarr = arr[..., idf]  # (time, space, space)
                iarr = iarr.reshape((iarr.shape[0], -1))  # (time, space)
                iarr = np.transpose(iarr, (1, 0))  # (space, time)
                _iout = Utilities.regrid_data(
                    iarr, self.regrid, self.hr_meta, target_meta, target_shape
                )
                _iarr, self.regrid = _iout
                _iarr = np.expand_dims(_iarr, -1)
                regridded_arr.append(_iarr)
            arr = np.concatenate(regridded_arr, -1)

        if len(arr) == 0:
            arr = arr[0]
        elif len(arr.shape) == 4:
            arr = np.transpose(arr, (2, 0, 1, 3))

        return arr


class NetCDF:
    """netcdf i/o utilities"""

    @staticmethod
    def clean_encodings(encoding):
        """Clean any bad encoding values e.g.:
        - integer scale factors
        """
        for fname, en_attrs in encoding.items():
            for aname, value in en_attrs.items():
                if isinstance(value, int):
                    encoding[fname][aname] = float(value)
        return encoding

    @staticmethod
    def convert_time_index(time_index):
        """Convert a pandas datetimeindex to CF standard

        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index object with meta data about time axis

        Returns
        -------
        values : np.ndarray
            1D array of time units in either hours or minutes since
            1970-01-01 00:00
        units : str
            Units (either hour or minutes since 1970-01-01 00:00)
        """

        if isinstance(time_index, pd.Timestamp):
            time_index = pd.to_datetime([time_index])

        if isinstance(time_index, pd.DatetimeIndex):
            time_index = time_index.astype(str)

        time = [parser.parse(t) for t in time_index]

        units = 'minutes since 1970-01-01 00:00'
        if len(time) > 1:
            diff = time[1] - time[0]
            if diff.seconds == 3600:
                units = 'hours since 1970-01-01 00:00'

        values = date2num(time, units)

        return values, units

    @classmethod
    def make_dataarray(cls, array, time_index, latitude, longitude, attrs):
        """Make an xarray data array object

        Parameters
        ----------
        array : np.ndarray
            Spatiotemporal 3D array of shape (time, latitude, longitude)
            or 2D with shape (latitude, longitude)
        time_index : pd.DatetimeIndex
            Time index object with meta data about time axis
        latitude : np.ndarray
            1D array of latitude values in decimal degrees
        longitude : np.ndarray
            1D array of longitude values in decimal degrees
        attrs : dict
            Variable attributes including standard_name,
            long_name, units, description

        Returns
        -------
        darray : xarray.DataArray
        """

        # Get the time index and it's units
        time, time_units = cls.convert_time_index(time_index)

        # Build Data Array
        if len(array.shape) == 3:
            coords = [time, latitude, longitude]
            dims = ['time', 'latitude', 'longitude']
        elif len(array.shape) == 2:
            coords = [latitude, longitude]
            dims = ['latitude', 'longitude']
        else:
            raise ValueError(f'Bad array shape: {array.shape}')

        dtype = attrs.get('dtype', 'float32')
        if 'float' in dtype:
            dtype_max = np.finfo(attrs['dtype']).max
        else:
            dtype_max = np.iinfo(attrs['dtype']).max

        darray = array.astype(np.float32)
        darray = np.maximum(darray, attrs['valid_min'])
        darray = np.minimum(darray, attrs['valid_max'])
        darray = xr.DataArray(darray, coords=coords, dims=dims)

        darray.attrs['standard_name'] = attrs['standard_name']
        darray.attrs['long_name'] = attrs['long_name']
        darray.attrs['units'] = attrs['units']
        darray.attrs['description'] = attrs['description']
        darray.attrs['missing_value'] = dtype_max
        darray.attrs['_FillValue'] = dtype_max
        darray.attrs['valid_min'] = attrs['valid_min']
        darray.attrs['valid_max'] = attrs['valid_max']

        darray['latitude'].attrs['standard_name'] = 'latitude'
        darray['latitude'].attrs['long_name'] = 'latitude'
        darray['latitude'].attrs['units'] = 'degrees_north'

        darray['longitude'].attrs['standard_name'] = 'longitude'
        darray['longitude'].attrs['long_name'] = 'longitude'
        darray['longitude'].attrs['units'] = 'degrees_east'

        if dims[0] == 'time':
            darray['time'].encoding['units'] = time_units
            darray['time'].attrs['units'] = time_units
            darray['time'].attrs['standard_name'] = 'time'
            darray['time'].attrs['long_name'] = 'time'

        darray.attrs['grid_mapping'] = 'crs'

        return darray

    @classmethod
    def make_dataset(cls, data_vars, fp_out=None, encoding=None):
        """Make an xarray dataset object

        Parameters
        ----------
        data_vars : dict
            Dictionary of variables to write to netcdf with format
            {name: darray} where name is a string and darray is
            output from cls.make_dataarray
        fp_out : str
            Option to write xr dataset to .nc file on disk
        encoding : dict | None
            optional output file enconding, e.g.,
            {"my_variable": {"dtype": "int16", "scale_factor": 0.1}}

        Returns
        -------
        dset : xarray.Dataset
        """

        encoding = cls.clean_encodings(encoding)

        # Build Dataset
        ds = xr.Dataset(data_vars=data_vars)

        # Add coordinate reference object
        ds['crs'] = int()  # noqa: UP018
        ds['crs'].attrs['long_name'] = 'coordinate reference system'
        ds['crs'].attrs['grid_mapping_name'] = 'latitude_longitude'
        ds['crs'].attrs['longitude_of_prime_meridian'] = 0.0
        ds['crs'].attrs['semi_major_axis'] = 6378137.0
        ds['crs'].attrs['inverse_flattening'] = 298.257223563
        ds['crs'].attrs['crs_wkt'] = (
            'GEOGCS["WGS 84",\nDATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.'
            '257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],\n'
            'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],\nUNIT["degree",'
            '0.01745329251994328,AUTHORITY["EPSG","9122"]],\nAUTHORITY'
            '["EPSG","4326"]]'
        )
        ds['crs'].attrs['proj4_params'] = (
            '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        )
        ds['crs'].attrs['epsg_code'] = 'EPSG:4326'

        # Global Attributes, CF-1.7 may not be latest
        ds.attrs['Conventions'] = 'CF-1.7'
        ds.attrs['title'] = 'Urban heat island data for sup3r'
        ds.attrs['nc.institution'] = 'Unidata'
        ds.attrs['source'] = 'Sup3rUHI'
        ds.attrs['date'] = str(datetime.datetime.utcnow())
        ds.attrs['references'] = ''
        ds.attrs['comment'] = ''

        # Write to file
        if fp_out is not None:
            fp_out_tmp = fp_out + '.tmp'
            ds.load().to_netcdf(
                fp_out_tmp,
                format='NETCDF4',
                engine='h5netcdf',
                encoding=encoding,
            )
            shutil.move(fp_out_tmp, fp_out)

        return ds
