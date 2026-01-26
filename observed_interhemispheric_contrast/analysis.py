import cmaps
import calendar
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, LogNorm, PowerNorm

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'  
mpl.rcParams['font.size'] = 18


# select TC data
def IBTrACS_Select_TS(AGENCY='USA'):
    data = pd.read_csv('/mnt/h/Data/IBTrACS/ibtracs.ALL.list.v04r01.csv', keep_default_na=False) 
   
    data = data[data['NATURE'].isin(['TS'])]  
    
    columns = ['SID', 'SEASON', 'BASIN', 'NAME', 'ISO_TIME', AGENCY + '_LAT', AGENCY + '_LON', AGENCY + '_WIND', AGENCY + '_PRES']
    if AGENCY == 'USA':
        columns.append(AGENCY + '_SSHS')
    elif AGENCY == 'CMA':
        columns.append(AGENCY + '_CAT')
    elif AGENCY == 'TOKYO':
        columns.append(AGENCY + '_GRADE')
    data = data[columns]
    
    data['SEASON'] = data['SEASON'].astype('int')
    data = data[data['SEASON'].isin(range(1980, 2021))] # Select years

    data = data[~data['BASIN'].isin(['SA'])]
    data['BASIN'] = data['BASIN'].replace({'WP': 'WNP', 'EP': 'ENP'})

    data = data[data['NAME'] != 'UNNAMED']    # Select named TCs
    
    data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME']) 
    data = data[data['ISO_TIME'].dt.hour.isin([0, 6, 12, 18])]
  
    data = data[data[AGENCY + '_WIND'] != ' ']

    data['SID'] = data['SID'].astype('string')
    data['BASIN'] = data['BASIN'].astype('string')
    data['NAME'] = data['NAME'].astype('string')
    data[AGENCY + '_LAT'] = data[AGENCY + '_LAT'].astype('float')
    data[AGENCY + '_LON'] = data[AGENCY + '_LON'].astype('float')
    
    return data

JTWC_data = IBTrACS_Select_TS(AGENCY='USA')



# basin statistics
data_TCF = JTWC_data.copy()
data_TCF = data_TCF.drop_duplicates(subset='SID', keep='first')
data_TCF['USA_LON'] = data_TCF['USA_LON'].apply(lambda x: x if x >= 0 else x + 360)

data_TCF['Hemisphere'] = data_TCF['USA_LAT'].apply(lambda x: 'NH' if x > 0 else 'SH')
annual_counts = data_TCF.groupby('SEASON')['SID'].nunique().reset_index()
annual_counts.columns = ['SEASON', 'All']
NH_counts = data_TCF[data_TCF['Hemisphere'] == 'NH'].groupby('SEASON')['SID'].nunique().reset_index()
NH_counts.columns = ['SEASON', 'NH']
south_counts = data_TCF[data_TCF['Hemisphere'] == 'SH'].groupby('SEASON')['SID'].nunique().reset_index()
south_counts.columns = ['SEASON', 'SH']
basin_counts = data_TCF.groupby(['SEASON', 'BASIN'])['SID'].nunique().unstack(fill_value=0).reset_index()

result = annual_counts.merge(NH_counts, on='SEASON', how='left')
result = result.merge(south_counts, on='SEASON', how='left')
result = result.merge(basin_counts, on='SEASON', how='left')
result = result[['SEASON', 'All', 'NH', 'SH', 'WNP', 'ENP', 'NA', 'SI', 'SP', 'NI']]
result

# save time series of TCF
# result.to_csv('H:/processedData/TimeSeries_TCF.csv', index=False)


# calculate monthly TCF
lat_bins = np.arange(91.25, -91.25, -2.5)   
lon_bins = np.arange(-1.25, 361, 2.5) 

monthly_data = []
for year in sorted(data_TCF['SEASON'].unique()):
    for month in sorted(data_TCF['MONTH'].unique()):
        data_month = data_TCF[(data_TCF['SEASON'] == year) & (data_TCF['MONTH'] == month)]
        res = pd.DataFrame(index=lat_bins[:-1], columns=lon_bins[:-1], data=0)
        
        for tc_id in data_month['SID'].unique():
            data_tc_first = data_month[data_month['SID'] == tc_id].iloc[0]
            lat = data_tc_first['USA_LAT']
            lon = data_tc_first['USA_LON']
            lat_index = np.digitize(lat, lat_bins) - 1
            lon_index = np.digitize(lon, lon_bins) - 1

            if (0 <= lat_index < len(lat_bins) - 1) and (0 <= lon_index < len(lon_bins) - 1):
                lat_grid = lat_bins[lat_index]
                lon_grid = lon_bins[lon_index]
                res.loc[lat_grid, lon_grid] += 1

        da = xr.DataArray(
            res.values,
            dims=['lat', 'lon'],
            coords={
                'lat': lat_bins[:-1] + 1.25,  
                'lon': lon_bins[:-1] + 1.25,   
            }
        )
        monthly_data.append(da)

Monthly_TCF_2p5 = xr.concat(monthly_data, dim='time')
Monthly_TCF_2p5 = Monthly_TCF_2p5.assign_coords(time=pd.date_range(start='1980-01', periods=492, freq='MS'))

# save Monthly TCF
Monthly_TCF_2p5.name = 'tcf'
Monthly_TCF_2p5.attrs['name'] = 'monthly tropical cyclone frequency'
Monthly_TCF_2p5.attrs['units'] = 'unitless'
Monthly_TCF_2p5 = Monthly_TCF_2p5.astype(np.float32)
Monthly_TCF_2p5.coords['lat'] = np.arange(88.75, -88.75 - 2.5, -2.5)
Monthly_TCF_2p5.coords['lon'] = np.arange(1.25, 358.75 + 2.5, 2.5)
# Monthly_TCF_2p5.to_netcdf('H:/ProcessedData/Monthly_TCF_2p5.nc')

# calculate annual TCF
years = Monthly_TCF_2p5.time.dt.year
months = Monthly_TCF_2p5.time.dt.month
Annual_TCF_2p5 = []
for year in np.unique(years):
    year_data = Monthly_TCF_2p5.sel(time=Monthly_TCF_2p5.time.dt.year == year)
    
    # NH
    nh_mask = (year_data.lat > 0) & (year_data.time.dt.month.isin([6, 7, 8, 9, 10, 11]))
    nh_data = year_data.where(nh_mask, drop=True)
    
    # SH
    sh_mask = (year_data.lat < 0) & (year_data.time.dt.month.isin([12, 1, 2, 3, 4, 5]))
    sh_data = year_data.where(sh_mask, drop=True)
    
    combined_data = xr.concat([nh_data, sh_data], dim='lat').sortby('lat', ascending=False)
    yearly_avg = combined_data.sum(dim='time', keep_attrs=True)
    yearly_avg = yearly_avg.assign_coords(year=year) 
    Annual_TCF_2p5.append(yearly_avg)

# save annual TCF
Annual_TCF_2p5 = xr.concat(Annual_TCF_2p5, dim='year')
Annual_TCF_2p5['year'] = pd.to_datetime(np.unique(years), format='%Y')
Annual_TCF_2p5 = Annual_TCF_2p5.rename({'year': 'time'})
Annual_TCF_2p5.attrs['name'] = 'annual tropical cyclone frequency'
# Annual_TCF_2p5.to_netcdf('/mnt/h/ProcessedData/Annual_TCF_2p5.nc')

# calculate TCF spatial trends
trend = np.zeros_like(Annual_TCF_2p5.isel(time=0))
p_value = np.zeros_like(Annual_TCF_2p5.isel(time=0))
for lat in range(len(Annual_TCF_2p5.lat)):
    for lon in range(len(Annual_TCF_2p5.lon)):
        y = Annual_TCF_2p5[:, lat, lon].values
        res = stats.linregress(np.arange(len(Annual_TCF_2p5.time)), y)
        trend[lat, lon] = res.slope * 10
        p_value[lat, lon] = res.pvalue
trend = np.where(trend == 0, np.nan, trend)


