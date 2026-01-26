import gc
import glob
import cmaps
import numpy as np 
import pandas as pd 
import xesmf as xe
import xarray as xr
import netCDF4 as nc
import metpy.calc as mpcalc 
from metpy.units import units 
from scipy import stats, linalg
from sklearn.linear_model import LinearRegression
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
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



# read and regrid DAMIP data

BASE_PATH = '/mnt/h/Data/CMIP6/DAMIP'
OUTPUT_PATH = '/mnt/h/ProcessedData/DAMIP'
MODELS = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 'FGOALS-g3', 'GFDL-ESM4',
          'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM']
EXPERIMENTS = {
    'piControl': {'time_slice': None, 'ssp245': False},
    'historical': {'time_slice': slice('1980', '2020'), 'ssp245': True},
    'hist-GHG': {'time_slice': slice('1980', '2020'), 'ssp245': False},
    'hist-aer': {'time_slice': slice('1980', '2020'), 'ssp245': False},
    'hist-nat': {'time_slice': slice('1980', '2020'), 'ssp245': False}
}
VARIABLES = {
    'u': {'var_name': 'ua', 'plevs': [85000, 50000, 20000], 'attrs': {'name': 'u component of wind', 'units': 'm/s'}},
    'v': {'var_name': 'va', 'plevs': [85000, 20000], 'attrs': {'name': 'v component of wind', 'units': 'm/s'}},
    'w': {'var_name': 'wap', 'plevs': [50000], 'attrs': {'name': 'vertical velocity at 500 hPa', 'units': 'Pa/s'}}
}
GRID_2P5 = {'lat': np.arange(91.25, -91.25, -2.5), 'lon': np.arange(-1.25, 361, 2.5)}


def regrid_model_experiment(model, experiment):
    print(f"Regridding {model} {experiment}...")
    exp_config = EXPERIMENTS[experiment]
 
    for var, var_config in VARIABLES.items():
        paths = f"{BASE_PATH}/{model}/{var_config['var_name']}_Amon_{model}_{experiment}_*.nc"
        files = sorted(glob.glob(paths))  
        
        if experiment == 'historical' and exp_config['ssp245']:
            ssp_paths = f"{BASE_PATH}/{model}/{var_config['var_name']}_Amon_{model}_ssp245_*.nc"
            ssp_files = sorted(glob.glob(ssp_paths))
            if xr.open_mfdataset(files, combine='by_coords').time[-1].dt.year <= 2014:
                files.append(ssp_files[0]) 
                ds = xr.open_mfdataset(files, combine='by_coords')
            else:
                historical_ds = xr.open_mfdataset(files, combine='by_coords')
                ssp_ds = xr.open_mfdataset(ssp_files, combine='by_coords')
                ssp_ds = ssp_ds.sel(time=slice(str(historical_ds.time[-1].item().year + 1), None))
                ds = xr.concat([historical_ds, ssp_ds], dim='time')
                historical_ds.close()
                ssp_ds.close()
        else:       
            ds = xr.open_mfdataset(files, combine='by_coords')

        if exp_config['time_slice'] is not None:
            ds = ds.sel(time=exp_config['time_slice'])
        
        grid_2p5 = xr.Dataset(coords={
            'time': ds.time,
            'lat': GRID_2P5['lat'],
            'lon': GRID_2P5['lon']
        })
        regridder = xe.Regridder(ds, grid_2p5, 'bilinear')
        var_data = regridder(ds[var_config['var_name']])
        var_data = var_data.astype(np.float32).isel(lat=slice(1, None), lon=slice(1, None))
        var_data = var_data.sel(plev=var_config['plevs'], method='nearest')
        var_data.name = var
        var_data.attrs.update(var_config['attrs'])
        
        # save regridded data
        output_file = f"{OUTPUT_PATH}/{model}_{experiment}_{var}_2p5.nc"
        print(f"Saving {output_file}...")
        var_data.to_netcdf(output_file)
        ds.close()
        
        del ds, regridder, var_data
        gc.collect()


def main():
    for model in MODELS:
        for experiment in EXPERIMENTS:
            regrid_model_experiment(model, experiment)
    print("Regridding complete")

if __name__ == "__main__":
    main()


    # calculate DGPI from model simulations

def calc_DGPI_from_model_experiment(model, experiment):
    u850 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_u_2p5.nc").u.sel(plev=85000, method='nearest')
    u500 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_u_2p5.nc").u.sel(plev=50000, method='nearest')
    u200 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_u_2p5.nc").u.sel(plev=20000, method='nearest')
    v850 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_v_2p5.nc").v.sel(plev=85000, method='nearest')
    v200 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_v_2p5.nc").v.sel(plev=20000, method='nearest')
    w500 = xr.open_dataset(f"/mnt/h/ProcessedData/DAMIP/{model}_{experiment}_w_2p5.nc").w.sel(plev=50000, method='nearest')

    # calculate VWS magnitude
    VWS = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)     

    # calculate dudy500
    dudy500 = u500.differentiate(coord='lat', edge_order=1)
    dudy500 = dudy500.where(dudy500.lat >= 0, -dudy500)       # reverse the SH
    dudy500 /= 111000                                         # convert [m/s/degree] to [1/s]

    # calculate AV850
    av850 = mpcalc.absolute_vorticity(u850, v850)             # in [1/s]

    # calculate DGPI
    coef = {
        'VWS'    :-1.7,
        'dudy500': 2.3,
        'w500'   : 3.4,
        'av850'  : 2.4
    }
    VWS_term = (2.0 + 0.1 * VWS) ** (coef['VWS'])
    dudy500_term = (5.5 - 1e5 * dudy500) ** (coef['dudy500'])
    w500_term = (5.0 - 20 * w500) ** (coef['w500'])
    av850_term = (5.5 + np.abs(1e5 * av850.values)) ** (coef['av850'])
    DGPI = VWS_term * dudy500_term * w500_term * av850_term * np.exp(-11.8) - 1.0
    DGPI = DGPI.drop_vars('plev')

    return DGPI


# calculate anomaly and regional average
def calc_annual_from_monthly(data):
    annual_data = []
    for year in np.unique(data.time.dt.year):
        data_year = data.sel(time=data.time.dt.year == year)
    
        nh_mask = (data_year.lat > 0) & (data_year.time.dt.month.isin([6, 7, 8, 9, 10, 11]))
        sh_mask = (data_year.lat < 0) & (data_year.time.dt.month.isin([12, 1, 2, 3, 4, 5]))
        
        combined_data = xr.concat([
            data_year.where(nh_mask, drop=True), 
            data_year.where(sh_mask, drop=True)
        ], dim='lat').sortby('lat', ascending=False)
            
        yearly_avg = combined_data.sum(dim='time', keep_attrs=True)
        yearly_avg = yearly_avg.assign_coords(year=year)
        annual_data.append(yearly_avg)
    
    annual_data = xr.concat(annual_data, dim='year')

    return annual_data


def calc_anomaly(data, start_year='1981', end_year='2010'):
    if data.year.size == 41:
        climatology = data.sel(year=slice(start_year, end_year)).mean(dim='year')
    else:
        climatology = data.mean(dim='year')
    data_anomaly = data - climatology
    return data_anomaly


def calc_region_average(data_anomaly, region='global'):
    MDRs = [
        {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
        {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
        {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
        {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -20, 'lat_max': -5},
        {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
        {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20}
    ]
    
    if region == 'global':
        region_average = []
        for r in MDRs:
            region_data = data_anomaly.sel(
                lon=slice(r['lon_min'], r['lon_max']), 
                lat=slice(r['lat_max'], r['lat_min'])
            ).mean(dim=['lon', 'lat'])
            region_average.append(region_data) 
        region_data = sum(region_average) / len(region_average)
        
    else:
        region_info = next((r for r in MDRs if r['name'] == region), None)
        if region_info is None:
            raise ValueError('Undefined region')
        
        region_data = data_anomaly.sel(
            lon=slice(region_info['lon_min'], region_info['lon_max']), 
            lat=slice(region_info['lat_max'], region_info['lat_min'])
        ).mean(dim=['lon', 'lat'])
    
    return region_data


# read monthly TCF observations
Monthly_TCF_2p5 = xr.open_dataset('/mnt/h/ProcessedData/Monthly_TCF_2p5.nc').tcf
TCF_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(Monthly_TCF_2p5)), region='global')

# read monthly DGPI observations
Monthly_DGPI_2p5 = xr.open_dataset('/mnt/h/ProcessedData/DGPI_2p5.nc').dgpi
DGPI_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(Monthly_DGPI_2p5)), region='global')

pictrl_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(DGPI_pictrl)), region='global')
hist_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(DGPI_hist)), region='global')
aer_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(DGPI_aer)), region='global')
ghg_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(DGPI_ghg)), region='global')
nat_global = calc_region_average(calc_anomaly(calc_annual_from_monthly(DGPI_nat)), region='global')


# perform optimal fingerprinting using OLS and TLS
def optimal_fingerprinting(y, X, pictrl, reg_method, n_boot=1000, ci_range=(2.5, 97.5), random_seed=None):
    # input processing
    y = y.values if isinstance(y, xr.DataArray) else y
    X = X.values if isinstance(X, xr.DataArray) else X
    X = X.reshape(-1, 1) if X.ndim == 1 else X

    if random_seed is not None:
        np.random.seed(random_seed)
    
    if reg_method == 'OLS':
        beta = LinearRegression(fit_intercept=False).fit(X, y).coef_
    
    elif reg_method == 'TLS':
        Z = np.column_stack([X, y])
        _, _, Vh = linalg.svd(Z, full_matrices=False)
        V = Vh.T
        beta = (-V[:X.shape[1], X.shape[1]:] / V[X.shape[1]:, X.shape[1]:]).flatten()
    
    results = {'beta': beta}

    # bootstrap CIs
    pictrl = pictrl.values if isinstance(pictrl, xr.DataArray) else pictrl

    boot_beta = np.zeros([n_boot, X.shape[1]])
    for i in range(n_boot):
        y_boot = y + np.random.choice(pictrl, size=len(y))
        if reg_method == 'OLS':
            boot_beta[i] = LinearRegression(fit_intercept=False).fit(X, y_boot).coef_
        else:
            Z_boot = np.column_stack([X, y_boot])
            _, _, Vh_boot = linalg.svd(Z_boot, full_matrices=False)
            V_boot = Vh_boot.T
            boot_beta[i] = (-V_boot[:X.shape[1], X.shape[1]:] / V_boot[X.shape[1]:, X.shape[1]:]).flatten()
    
    results.update({
        'ci_lower': np.percentile(boot_beta, ci_range[0], axis=0),
        'ci_upper': np.percentile(boot_beta, ci_range[1], axis=0),
        'p_values': np.minimum(2 * np.minimum(
            (boot_beta < 0).mean(axis=0),
            (boot_beta > 0).mean(axis=0)), 1)
    })
    
    return results


def print_results(results, reg_method):
    print(f"\nOptimal fingerprinting using {reg_method}")
    print(f"{'forcing':<15} {'beta':>10} {'CI[5-95%]':>20} {'p-value':>10}")
    print('-' * 60)
    for i, beta in enumerate(results['beta']):
        ci_lower = results.get('ci_lower', [np.nan] * len(results['beta']))[i]
        ci_upper = results.get('ci_upper', [np.nan] * len(results['beta']))[i]
        p_value = results.get('p_values', [np.nan] * len(results['beta']))[i]
        
        ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]" if not np.isnan(ci_lower) else "N/A"
        p_str = f"{p_value:.3f}" if not np.isnan(p_value) else "N/A"
        print(f"{f'fingerprint {i+1}':<15} {beta:>10.2f} {ci_str:>20} {p_str:>10}")



