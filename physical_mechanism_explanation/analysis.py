import numpy as np  
import xesmf as xe
import xarray as xr
import pandas as pd
import netCDF4 as nc
import pymannkendall as mk
import statsmodels.api as sm
import metpy.calc as mpcalc 
from metpy.units import units 
from scipy import stats, interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial' 
mpl.rcParams['font.size'] = 18


# read and process data
sat = xr.open_dataset('/mnt/h/ProcessedData/T2m_2p5.nc').t2m
u850= xr.open_dataset('/mnt/h/ProcessedData/u850_2p5.nc').u850
v850 = xr.open_dataset('/mnt/h/ProcessedData/v850_2p5.nc').v850
u200 = xr.open_dataset('/mnt/h/ProcessedData/u200_2p5.nc').u200
v200 = xr.open_dataset('/mnt/h/ProcessedData/v200_2p5.nc').v200
vws = xr.open_dataset('/mnt/h/ProcessedData/VWS_2p5.nc').vws / 2

U = xr.open_dataset('/mnt/h/ProcessedData/U_2p5.nc').u
U_nh = U.where(U['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True).mean('lon')
U_sh = U.where(U['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True).mean('lon')
delta_U_nh = U_nh.sel(time=slice('2001', '2020')).mean('time') - U_nh.sel(time=slice('1980', '2000')).mean('time')
delta_U_sh = U_sh.sel(time=slice('2001', '2020')).mean('time') - U_sh.sel(time=slice('1980', '2000')).mean('time')

V = xr.open_dataset('/mnt/h/ProcessedData/V_2p5.nc').v
V_nh = V.where(V['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True).mean('lon')
V_sh = V.where(V['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True).mean('lon')
delta_V_nh = V_nh.sel(time=slice('2001', '2020')).mean('time') - V_nh.sel(time=slice('1980', '2000')).mean('time')
delta_V_sh = V_sh.sel(time=slice('2001', '2020')).mean('time') - V_sh.sel(time=slice('1980', '2000')).mean('time')

W = xr.open_dataset('/mnt/h/ProcessedData/W_2p5.nc').w
W_nh = W.where(W['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True).mean('lon')
W_sh = W.where(W['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True).mean('lon')
delta_W_nh = W_nh.sel(time=slice('2001', '2020')).mean('time') - W_nh.sel(time=slice('1980', '2000')).mean('time')
delta_W_sh = W_sh.sel(time=slice('2001', '2020')).mean('time') - W_sh.sel(time=slice('1980', '2000')).mean('time')

MDRs = [
    {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20},  
]