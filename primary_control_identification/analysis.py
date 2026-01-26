import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from itertools import combinations
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skexplain import ExplainToolkit
from sklearn.ensemble import RandomForestRegressor
from alibi.explainers import PartialDependence, plot_pd, ALE, plot_ale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'  
mpl.rcParams['font.size'] = 18



# data preparation
annual_TCF = pd.read_csv('/mnt/h/processedData/TimeSeries_TCF.csv')

# 5 atmospheric data
RH600 = xr.open_dataset('/mnt/h/processedData/rh600_2p5.nc').rh600
RV850 = xr.open_dataset('/mnt/h/processedData/rv850_2p5.nc').rv850
AV850 = xr.open_dataset('/mnt/h/processedData/av850_2p5.nc').av850
VWS = xr.open_dataset('/mnt/h/processedData/VWS_2p5.nc').vws
W500 = xr.open_dataset('/mnt/h/processedData/w500_2p5.nc').w500

# 6 oceanic data
SST = xr.open_dataset('/mnt/h/processedData/SST_2p5.nc').sst
SSS = xr.open_dataset('/mnt/h/processedData/SSS_2p5.nc').sss
MLD = xr.open_dataset('/mnt/h/processedData/MLDp03_2p5.nc').mldp03
D26 = xr.open_dataset('/mnt/h/processedData/D26_2p5.nc').d26
T100 = xr.open_dataset('/mnt/h/processedData/T100_2p5.nc').t100
TCHP = xr.open_dataset('/mnt/h/processedData/TCHP_2p5.nc').tchp

# MPI
MPI = xr.open_dataset('/mnt/h/processedData/MPI_2p5.nc').mpi

# define MDRs
MDRs = {
    'WNP': {'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    'ENP': {'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    'NA': {'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    'SI': {'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    'SP': {'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    'NI': {'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20}
}

# select data in MDRs (6–11 for NH and 12–5 for SH)
def sel_mdr_data(dataset, mdr_name):
    mdr = MDRs[mdr_name]  
    if mdr['lat_min'] > 0:
        data_mdr = dataset.sel(
            lon=slice(mdr['lon_min'], mdr['lon_max']),
            lat=slice(mdr['lat_max'], mdr['lat_min']),
        )
        data_mdr = data_mdr.where(data_mdr['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
    else: 
        data_mdr = dataset.sel(
            lon=slice(mdr['lon_min'], mdr['lon_max']),
            lat=slice(mdr['lat_max'], mdr['lat_min']),
        )
        data_mdr = data_mdr.where(data_mdr['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
    
    return data_mdr

variables = {
    'RH600': RH600,
    'RV850': RV850,
    'AV850': AV850,
    'VWS': VWS,
    'W500': W500,
    'SST': SST,
    'SSS': SSS,
    'MLD': MLD,
    'D26': D26,
    'T100': T100,
    'TCHP': TCHP,
    'MPI': MPI,
}

data_selected = {mdr: {} for mdr in MDRs.keys()}
for var_name, var_data in variables.items():
    for mdr_name in MDRs.keys():
        data_selected[mdr_name][var_name] = sel_mdr_data(var_data, mdr_name)

data_selected


# data merging
merged_data = annual_TCF.copy()
merged_data = merged_data.rename(columns={'SEASON': 'year'})

for mdr_name in data_selected.keys():
    for var_name in data_selected[mdr_name].keys():   
        ds = data_selected[mdr_name][var_name]
        if 'depth' in ds.dims:
            annual_mean = ds.mean(dim=['lat', 'lon', 'depth']).groupby('time.year').mean()
        else:
            annual_mean = ds.mean(dim=['lat', 'lon']).groupby('time.year').mean()
        annual_mean_df = annual_mean.to_dataframe(name=f'{mdr_name}_{var_name}').reset_index()
    
        merged_data = merged_data.merge(
            annual_mean_df, 
            left_on='year', 
            right_on='year', 
            how='left',
        )

for var in variables:
    cols = [f'{basin}_{var}' for basin in ['WNP', 'ENP', 'NA', 'SI', 'SP', 'NI']]
    
    # reverse the sign of AV850 and RV850 in SH
    if var in ['AV850', 'RV850']:    
        weights = [1, 1, 1, -1, -1, 1]
        weighted_data = merged_data[cols] * weights
        merged_data[f'ALL_{var}'] = weighted_data.mean(axis=1)
    else:
        merged_data[f'ALL_{var}'] = merged_data[cols].mean(axis=1)

merged_data

# save data
# merged_data.to_csv('H:/ProcessedData/IML_data.csv', index=False)


# define the evaluation function
def evaluate_model(model, X, y, name='Test'):
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    
    print(f'{name} set metrics:')
    print(f'R2: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.4f}%')

    plt.figure(figsize=(2, 2))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([60, 100], [60, 100], 'k--', lw=2)
    plt.xlim([60, 100])
    plt.xticks([60, 80, 100])
    plt.ylim([60, 100])
    plt.yticks([60, 80, 100])
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.title(f'{name} set')
    plt.show()


    # random forest for global-scale analysis
X = merged_data.filter(regex='ALL_')
y = merged_data['All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1577)

rf = RandomForestRegressor(
    n_estimators=50,
    criterion='friedman_mse', 
    max_depth=5, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=1.0, 
    bootstrap=True, 
    n_jobs=-1, 
    random_state=1983, 
    max_samples=1.0, 
)

rf_params = {
    'n_estimators': range(20, 26),
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
}
rf_gs = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='r2',
)
rf_gs.fit(X_train, y_train)
best_rf_params = rf_gs.best_params_
best_rf = rf_gs.best_estimator_

evaluate_model(best_rf, X_train, y_train, name='Training')
evaluate_model(best_rf, X_test, y_test, name='Test')

pd.Series(best_rf.feature_importances_, index=X.columns).sort_values().tail(12).plot(kind='barh')
plt.title('Random forest feature importance')
plt.show()


# XGBoost for global-scale analysis
X = merged_data.filter(regex='ALL_')
y = merged_data['All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1577)

xgb = XGBRegressor(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    min_child_weight=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=1577,  
)

xgb_params = {
    'n_estimators': range(5, 15),
    'max_depth': [5, 10, 15],
    'learning_rate': [0.3, 0.4, 0.5],
    'reg_alpha': [0.5, 1.0, 1.5],
    'reg_lambda': [0.4, 0.6, 0.8],  
}
xgb_gs = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='r2',
)
xgb_gs.fit(X_train, y_train)
best_xgb_params = xgb_gs.best_params_
best_xgb = xgb_gs.best_estimator_

evaluate_model(best_xgb, X_train, y_train, name='Training')
evaluate_model(best_xgb, X_test, y_test, name='Test')

pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values().tail(12).plot(kind='barh')
plt.title('XGBoost feature importance')
plt.show()


# LightGBM for global-scale analysis
X = merged_data.filter(regex='ALL_')
y = merged_data['All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1577)

lgb = LGBMRegressor(
    num_leaves=5,
    max_depth=10,
    learning_rate=0.1,  
    n_estimators=50,
    min_child_weight=1.0,
    min_child_samples=4,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.8,
    reg_lambda=0.8,
    random_state=1577,  
    n_jobs=-1,
    verbose=-1,
)

lgb_params = {
    'num_leaves': [5],
    'max_depth': [5],
    'learning_rate': [0.1],
    'n_estimators': [20],
    'reg_alpha': [0.8,],
    'reg_lambda': [1.2],  
}
lgb_gs = GridSearchCV(
    estimator=lgb,
    param_grid=lgb_params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='r2',
)
lgb_gs.fit(X_train, y_train)
best_lgb_params = lgb_gs.best_params_
best_lgb = lgb_gs.best_estimator_

evaluate_model(best_lgb, X_train, y_train, name='Training')
evaluate_model(best_lgb, X_test, y_test, name='Test')

pd.Series(best_lgb.feature_importances_, index=X.columns).sort_values().tail(12).plot(kind='barh')
plt.title('LightGBM feature importance')
plt.show()


