import shap
import cmaps
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


# Global
X = merged_data.filter(regex='ALL_')
y = merged_data['All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1577)

fig = plt.figure(figsize=(26, 16.5), dpi=300)
gs = gridspec.GridSpec(3, 5, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1]) 
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax5 = plt.subplot(gs[1, 0])
ax6 = plt.subplot(gs[1, 1])
ax7 = plt.subplot(gs[1, 2])
ax8 = plt.subplot(gs[2, 0])
ax9 = plt.subplot(gs[2, 1])
ax10 = plt.subplot(gs[2, 2])


y_train_rf = best_model.predict(X_train)
y_test_rf = best_model.predict(X_test)
ax1.scatter(y_train, y_train_rf, marker='o', s=150, c='#0d7ca1', alpha=0.8, ec='k', lw=1.5, label='Training set')
ax1.scatter(y_test, y_test_rf, marker='o', s=150, c='#ec721b', alpha=0.8, ec='k', lw=1.5, label='Test set')
ax1.plot([60, 100], [60, 100], 'k--', lw=1.5, alpha=0.5)
ax1.set_xlim([60, 100])
ax1.set_ylim([60, 100])
ax1.set_yticks([60, 70, 80, 90, 100])
ax1.set_xlabel('Observed TCF', fontsize=26, labelpad=8)
ax1.set_ylabel('Predicted TCF', fontsize=26)
ax1.tick_params(axis='both', labelsize=24)
ax1.text(0.03, 0.86, r'$\mathrm{Training}$', fontsize=22, color='#0d7ca1', transform=ax1.transAxes, ha='left', va='bottom')
ax1.text(0.03, 0.76, r'$R^2=0.76$', fontsize=22, color='#0d7ca1', transform=ax1.transAxes, ha='left', va='bottom')
ax1.text(0.03, 0.66, r'$\mathrm{RMSE=4.22}$', fontsize=22, color='#0d7ca1', transform=ax1.transAxes, ha='left', va='bottom')
ax1.text(0.965, 0.25, r'$\mathrm{Test}$', fontsize=22, color='#ec721b', transform=ax1.transAxes, ha='right', va='bottom')
ax1.text(0.97, 0.15, r'$R^2=0.67$', fontsize=22, color='#ec721b', transform=ax1.transAxes, ha='right', va='bottom')
ax1.text(0.97, 0.05, r'$\mathrm{RMSE=4.55}$', fontsize=22, color='#ec721b', transform=ax1.transAxes, ha='right', va='bottom')
ax1.set_title(r'$\mathbf{a}$', fontsize=28, loc='left')
ax1.set_title('RF', fontsize=28)
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

    
y_train_xgb = best_model.predict(X_train)
y_test_xgb = best_model.predict(X_test)
ax2.scatter(y_train, y_train_xgb, marker='o', s=150, c='#0d7ca1', alpha=0.8, ec='k', lw=1.5, label='Training set')
ax2.scatter(y_test, y_test_xgb, marker='o', s=150, c='#ec721b', alpha=0.8, ec='k', lw=1.5, label='Test set')
ax2.plot([60, 100], [60, 100], 'k--', lw=1.5, alpha=0.5)
ax2.set_xlim([60, 100])
ax2.set_ylim([60, 100])
ax2.set_yticks([60, 70, 80, 90, 100])
ax2.set_xlabel('Observed TCF', fontsize=26, labelpad=8)
ax2.tick_params(axis='both', labelsize=24)
ax2.text(0.03, 0.86, r'$\mathrm{Training}$', fontsize=22, color='#0d7ca1', transform=ax2.transAxes, ha='left', va='bottom')
ax2.text(0.03, 0.76, r'$R^2=0.89$', fontsize=22, color='#0d7ca1', transform=ax2.transAxes, ha='left', va='bottom')
ax2.text(0.03, 0.66, r'$\mathrm{RMSE=2.74}$', fontsize=22, color='#0d7ca1', transform=ax2.transAxes, ha='left', va='bottom')
ax2.text(0.965, 0.25, r'$\mathrm{Test}$', fontsize=22, color='#ec721b', transform=ax2.transAxes, ha='right', va='bottom')
ax2.text(0.97, 0.15, r'$R^2=0.78$', fontsize=22, color='#ec721b', transform=ax2.transAxes, ha='right', va='bottom')
ax2.text(0.97, 0.05, r'$\mathrm{RMSE=3.79}$', fontsize=22, color='#ec721b', transform=ax2.transAxes, ha='right', va='bottom')
ax2.set_title(r'$\mathbf{b}$', fontsize=28, loc='left')
ax2.set_title('XGBoost', fontsize=28)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


y_train_lgb = best_model.predict(X_train)
y_test_lgb = best_model.predict(X_test)
ax3.scatter(y_train, y_train_lgb, marker='o', s=150, c='#0d7ca1', alpha=0.8, ec='k', lw=1.5, label='Training set')
ax3.scatter(y_test, y_test_lgb, marker='o', s=150, c='#ec721b', alpha=0.8, ec='k', lw=1.5, label='Test set')
ax3.plot([60, 100], [60, 100], 'k--', lw=1.5, alpha=0.5)
ax3.set_xlim([60, 100])
ax3.set_ylim([60, 100])
ax3.set_yticks([60, 70, 80, 90, 100])
ax3.set_xlabel('Observed TCF', fontsize=26, labelpad=8)
ax3.tick_params(axis='both', labelsize=24)
ax3.text(0.03, 0.86, r'$\mathrm{Training}$', fontsize=22, color='#0d7ca1', transform=ax3.transAxes, ha='left', va='bottom', weight='bold')
ax3.text(0.03, 0.76, r'$R^2=0.73$', fontsize=22, color='#0d7ca1', transform=ax3.transAxes, ha='left', va='bottom', weight='bold')
ax3.text(0.03, 0.66, r'$\mathrm{RMSE=4.29}$', fontsize=22, color='#0d7ca1', transform=ax3.transAxes, ha='left', va='bottom', weight='bold')
ax3.text(0.96, 0.25, r'$\mathrm{Test}$', fontsize=22, color='#ec721b', transform=ax3.transAxes, ha='right', va='bottom', weight='bold')
ax3.text(0.97, 0.15, r'$R^2=0.61$', fontsize=22, color='#ec721b', transform=ax3.transAxes, ha='right', va='bottom', weight='bold')
ax3.text(0.97, 0.05, r'$\mathrm{RMSE=5.05}$', fontsize=22, color='#ec721b', transform=ax3.transAxes, ha='right', va='bottom', weight='bold')
ax3.set_title(r'$\mathbf{c}$', fontsize=28, loc='left')
ax3.set_title('LightGBM', fontsize=28)
for spine in ax3.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
def calc_ci(ax, x, y, color, alpha=0.3):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_new = np.linspace(np.min(x), np.max(x), 100)
    y_pred = intercept + slope * x_new
    n = len(x)
    y_err = y - (intercept + slope * x)
    mean_x = np.mean(x)
    t = stats.t.ppf(1 - 0.05 / 2, df=n-2) 
    ci = t * np.sqrt(np.sum(y_err ** 2) / (n - 2)) * np.sqrt(1 / n + (x_new - mean_x) ** 2 / np.sum((x - mean_x) ** 2))
    ax.plot(x_new, y_pred, color=color, lw=2)
    ax.fill_between(x_new, y_pred - ci, y_pred + ci, color=color, alpha=alpha)
calc_ci(ax1, y_train, y_train_rf, '#0d7ca1')
calc_ci(ax1, y_test, y_test_rf, '#ec721b')
calc_ci(ax2, y_train, y_train_xgb, '#0d7ca1')
calc_ci(ax2, y_test, y_test_xgb, '#ec721b')
calc_ci(ax3, y_train, y_train_lgb, '#0d7ca1')
calc_ci(ax3, y_test, y_test_lgb, '#ec721b')


ax4 = fig.add_axes([ax7.get_position().x1 + 0.11, ax7.get_position().y0 + 0.014,
                    ax7.get_position().width * 1.88, ax7.get_position().height * 2 + 0.11])
explainer = shap.Explainer(best_model)
shap_values = explainer(X)
feature_names = ['TCHP', 'W500', 'MLD', 'VWS', 'SST', 'MPI', 'D26', 'T100', 'RV850', 'SSS', 'RH600', 'AV850']
beeswarm_plot = shap.plots.beeswarm(
    shap_values, 
    max_display=12, 
    show=False, 
    alpha=0.7, 
    s=100,
    color=cmaps.sunshine_diff_12lev,
    plot_size=None,
)
for collection in ax4.collections:
    collection.set_edgecolor('k') 
    collection.set_linewidth(1)  
ax4.set_xlim([-3, 9])
ax4.set_xlabel('')
ax4.set_yticklabels(feature_names[::-1])
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_tick_params(which='both', width=1.5, length=6)
ax4.yaxis.set_tick_params(which='both', width=1.5, length=6)
ax4.tick_params(axis='both', labelsize=24)
ax4.grid(True, which='both', axis='y', linestyle='--', lw=1.5, alpha=0.3)
ax4.set_title(r'$\mathbf{d}$', fontsize=28, loc='left')
ax4.set_title('SHAP value', fontsize=28)
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontsize(24)
for spine in ax4.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
if len(plt.gcf().axes) > 1:
    plt.gcf().axes[-1].remove()
cbar = fig.add_axes([0.816, 0.425, 0.015, 0.2])
sm = plt.cm.ScalarMappable(cmap=cmaps.sunshine_diff_12lev, norm=plt.Normalize(0, 1))
cbar = fig.colorbar(sm, cax=cbar, ticks=[0, 1])
cbar.ax.tick_params(axis='y', which='both', color='w', length=1.0)
cbar.ax.set_yticklabels(['Low', 'High'], fontsize=24)
cbar.set_label('Feature value', fontsize=26, labelpad=20, rotation=-90)
cbar.outline.set_linewidth(1.5) 
shap_values_abs_mean = np.abs(shap_values.values).mean(axis=0)
sorted_indices = np.argsort(shap_values_abs_mean)
sorted_values = shap_values_abs_mean[sorted_indices] 
ax44 = fig.add_axes([ax4.get_position().x0 + ax4.get_position().width, ax4.get_position().y0, 0.1828, ax4.get_position().height])
colors = ["#e43d33", "#e43d33", "#4977ba", "#e43d33", "#4977ba", "#4977ba",
          "#f5ba03", "#4977ba", "#e43d33", "#4977ba", "#e43d33", "#4977ba"]
bars = ax44.barh(feature_names, sorted_values, height=0.7, color=colors, alpha=0.8)
bars = ax44.barh(feature_names, sorted_values, height=0.7, color='none', ec='k', lw=1.5)
green_patch = mpatches.Patch(color='#e43d33', label='Atmospheric', ec='k', lw=1.5)
blue_patch = mpatches.Patch(color='#4977ba', label='Oceanic', ec='k', lw=1.5)
brown_patch = mpatches.Patch(color='#f5ba03', label='Integrated', ec='k', lw=1.5)
ax44.legend(frameon=False, fontsize=24, handles=[green_patch, blue_patch, brown_patch], bbox_to_anchor=(0.32, 0.18))
for bar in bars:
    width = bar.get_width()
    ax44.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f'$\ {width:.2f}$', ha='left', va='center', fontsize=22)
ax44.set_xlim([0, 2.5])
ax44.set_xticks([]) 
ax44.set_yticks([])
ax44.set_xticklabels('')
ax44.set_yticklabels('')
ax44.set_title(r'$\mathbf{e}$', fontsize=28, loc='left')
ax44.set_title('Feature importance', fontsize=28)
for spine in ax44.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
PD = explainer.pd(features='all', subsample=1.0, n_jobs=5, n_bins=8)
ICE = explainer.ice(features='all', subsample=1.0, n_jobs=5, n_bins=8)
for i in range(len(ICE.n_samples)):
    ax5.plot(ICE.NA_TCHP__bin_values, ICE.NA_TCHP__rf_NA__ice[i, :] + 78.95, color='grey', lw=0.5)
ax5.plot(PD.NA_TCHP__bin_values, PD.NA_TCHP__rf_NA__pd[0] + 78.95, color='k', lw=2.5)
bins = np.linspace(PD.NA_TCHP__bin_values.min(), PD.NA_TCHP__bin_values.max(), 11)  
hist, bin_edges = np.histogram(PD.NA_TCHP, bins=bins)
ax55 = ax5.twinx()
ax55.bar(bin_edges[:-1], hist, 
         width=(PD.NA_TCHP__bin_values.max() - PD.NA_TCHP__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax55.set_ylim([0, 20])
ax55.set_yticks([0, 5, 10, 15, 20])
ax55.set_yticklabels('')
ax55.tick_params(axis='y', labelsize=24)
ax5.set_xlim([PD.NA_TCHP__bin_values.min(), PD.NA_TCHP__bin_values.max()])
ax5.set_ylim([77.5, 80.5])
ax5.set_yticks(np.arange(77.5, 81, 0.5))
ax5.set_xticks(np.arange(15, 36, 5))
ax5.set_xticklabels(np.arange(45, 70, 5))
ax5.set_xlabel(r'$\mathrm{TCHP\ (kJ\ cm^{-2})}$', fontsize=26) 
ax5.set_ylabel('TCF (PDP & ICE)', fontsize=26) 
ax5.tick_params(axis='both', labelsize=24) 
ax5.set_zorder(ax55.get_zorder() + 1) 
ax5.set_title(r'$\mathbf{f}$', fontsize=28, loc='left')
ax5.patch.set_visible(False)  
for spine in ax5.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
PD = explainer.pd(features='all', subsample=1.0, n_jobs=5, n_bins=8)
ICE = explainer.ice(features='all', subsample=1.0, n_jobs=5, n_bins=8)
for i in range(len(ICE.n_samples)):
    ax6.plot(ICE.NA_W500__bin_values, ICE.NA_W500__rf_NA__ice[i, :] + 78.95, color='grey', lw=0.5)
ax6.plot(PD.NA_W500__bin_values, PD.NA_W500__rf_NA__pd[0] + 78.95, color='k', lw=2.5)
bins = np.linspace(PD.NA_W500__bin_values.min(), PD.NA_W500__bin_values.max(), 11)  
hist, bin_edges = np.histogram(PD.NA_W500, bins=bins)
ax66 = ax6.twinx()
ax66.bar(bin_edges[:-1], hist, 
         width=(PD.NA_W500__bin_values.max() - PD.NA_W500__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax66.set_ylim([0, 20])
ax66.set_yticks([0, 5, 10, 15, 20])
ax66.set_yticklabels('')
ax66.tick_params(axis='y', labelsize=24)
ax6.set_xlim([PD.NA_W500__bin_values.min(), PD.NA_W500__bin_values.max()])
ax6.set_xticks(np.arange(-0.022, -0.005, 0.004))
x_formatter = ScalarFormatter(useMathText=True)
x_formatter.set_powerlimits((0, 0)) 
ax6.xaxis.set_major_formatter(x_formatter)
ax6.xaxis.get_offset_text().set_fontsize(22)
ax6.xaxis.get_offset_text().set_position((1.17, 0))
ax6.xaxis.get_offset_text().set_verticalalignment('bottom')
ax6.set_ylim([77.5, 80.5])
ax6.set_yticks(np.arange(77.5, 81, 0.5))
ax6.set_yticklabels('')
ax6.set_xlabel(r'$\mathrm{W500\ (Pa\ s^{-1})}$', fontsize=26) 
ax6.tick_params(axis='both', labelsize=24)
ax6.set_zorder(ax66.get_zorder() + 1) 
ax6.patch.set_visible(False)  
ax6.set_title(r'$\mathbf{g}$', fontsize=28, loc='left')
for spine in ax6.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
PD = explainer.pd(features='all', subsample=1.0, n_jobs=5, n_bins=8)
ICE = explainer.ice(features='all', subsample=1.0, n_jobs=5, n_bins=8)
for i in range(len(ICE.n_samples)):
    ax7.plot(ICE.SI_MLD__bin_values, ICE.SI_MLD__rf_SI__ice[i, :] + 78.95, color='grey', lw=0.5)
ax7.plot(PD.SI_MLD__bin_values, PD.SI_MLD__rf_SI__pd[0] + 78.95, color='k', lw=2.5)
bins = np.linspace(PD.SI_MLD__bin_values.min(), PD.SI_MLD__bin_values.max(), 11)  
hist, bin_edges = np.histogram(PD.SI_MLD, bins=bins)
ax77 = ax7.twinx()
ax77.bar(bin_edges[:-1], hist, 
         width=(PD.SI_MLD__bin_values.max() - PD.SI_MLD__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax77.set_ylim([0, 20])
ax77.set_ylabel('Frequency', fontsize=26, rotation=-90, labelpad=30)
ax77.set_yticks([0, 5, 10, 15, 20])
ax77.tick_params(axis='y', labelsize=24)
ax7.set_xlim([PD.SI_MLD__bin_values.min(), PD.SI_MLD__bin_values.max()])
ax7.set_xticks(np.arange(22, 30, 2))
ax7.set_xticklabels(np.arange(24, 32, 2))
ax7.set_ylim([77.5, 80.5])
ax7.set_yticks(np.arange(77.5, 81, 0.5))
ax7.set_yticklabels('')
ax7.set_xlabel(r'$\mathrm{MLD\ (m)}$', fontsize=26) 
ax7.tick_params(axis='both', labelsize=24) 
ax7.set_zorder(ax77.get_zorder() + 1) 
ax7.patch.set_visible(False)  
ax7.set_title(r'$\mathbf{h}$', fontsize=28, loc='left')
for spine in ax7.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
ALE = explainer.ale(features='all', n_bootstrap=100, subsample=1.0, n_jobs=8, n_bins=4)
ALE_median = np.median(ALE.NA_TCHP__rf_NA__ale, axis=0)
ci_lower = np.percentile(ALE.NA_TCHP__rf_NA__ale, 5, axis=0)
ci_upper = np.percentile(ALE.NA_TCHP__rf_NA__ale, 95, axis=0)
ax8.plot(ALE.NA_TCHP__bin_values, ALE_median, color='r', lw=2.5)
ax8.fill_between(ALE.NA_TCHP__bin_values, ci_lower, ci_upper, fc='r', ec='w', alpha=0.4, lw=3.0)
ax8.axhline(y=0, color='grey', lw=1.5, linestyle='--')
bins = np.linspace(ALE.NA_TCHP__bin_values.min(), ALE.NA_TCHP__bin_values.max(), 11)  
hist, bin_edges = np.histogram(ALE.NA_TCHP, bins=bins)
ax88 = ax8.twinx()
ax88.bar(bin_edges[:-1], hist, 
         width=(ALE.NA_TCHP__bin_values.max() - ALE.NA_TCHP__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax88.set_ylim([0, 20])
ax88.tick_params(axis='y', labelsize=24)
ax88.set_yticklabels('')
ax8.set_xlim([ALE.NA_TCHP__bin_values.min(), ALE.NA_TCHP__bin_values.max()])
ax8.set_xticks(np.arange(17.5, 29.6, 3))
ax8.set_xticklabels(np.arange(48, 68, 4))
ax8.set_ylim([-1, 1])
ax8.set_xlabel(r'$\mathrm{TCHP\ (kJ\ cm^{-2})}$', fontsize=26) 
ax8.set_ylabel(r'$\mathrm{\Delta TCF\ (ALE)}$', fontsize=26) 
ax8.tick_params(axis='both', labelsize=24) 
ax8.set_zorder(ax88.get_zorder() + 1) 
ax8.set_title(r'$\mathbf{i}$', fontsize=28, loc='left')
ax8.patch.set_visible(False)  
for spine in ax8.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
ALE = explainer.ale(features='all', n_bootstrap=100, subsample=1.0, n_jobs=8, n_bins=4)
ALE_median = np.median(ALE.NA_W500__rf_NA__ale, axis=0)
ci_lower = np.percentile(ALE.NA_W500__rf_NA__ale, 5, axis=0)
ci_upper = np.percentile(ALE.NA_W500__rf_NA__ale, 95, axis=0)
ax9.plot(ALE.NA_W500__bin_values, ALE_median, color='r', lw=2.5)
ax9.fill_between(ALE.NA_W500__bin_values, ci_lower, ci_upper, fc='r', ec='w', alpha=0.4, lw=3.0)
ax9.axhline(y=0, color='grey', lw=1.5, linestyle='--')
bins = np.linspace(ALE.NA_W500__bin_values.min(), ALE.NA_W500__bin_values.max(), 11)  
hist, bin_edges = np.histogram(ALE.NA_W500, bins=bins)
ax99 = ax9.twinx()
ax99.bar(bin_edges[:-1], hist, 
         width=(ALE.NA_W500__bin_values.max() - ALE.NA_W500__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax99.set_ylim([0, 20])
ax99.tick_params(axis='y', labelsize=24)
ax99.set_yticklabels('')
ax9.set_xlim([ALE.NA_W500__bin_values.min(), ALE.NA_W500__bin_values.max()])
ax9.set_xticks(np.arange(-0.018, -0.009, 0.004))
x_formatter = ScalarFormatter(useMathText=True)
x_formatter.set_powerlimits((0, 0)) 
ax9.xaxis.set_major_formatter(x_formatter)
ax9.xaxis.get_offset_text().set_fontsize(22)
ax9.xaxis.get_offset_text().set_position((1.17, 0))
ax9.xaxis.get_offset_text().set_verticalalignment('bottom')
ax9.set_ylim([-1, 1])
ax9.set_yticklabels('')
ax9.set_xlabel(r'$\mathrm{W500\ (Pa\ s^{-1})}$', fontsize=26) 
ax9.tick_params(axis='both', labelsize=24) 
ax9.set_zorder(ax99.get_zorder() + 1) 
ax9.set_title(r'$\mathbf{j}$', fontsize=28, loc='left')
ax9.patch.set_visible(False)  
for spine in ax9.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


X = merged_data.filter(regex='ALL_')
y = merged_data['ALL']
explainer = ExplainToolkit(('rf_ALL', best_rf), X=X, y=y)
ALE = explainer.ale(features='all', n_bootstrap=100, subsample=1.0, n_jobs=8, n_bins=4)
ALE_median = np.median(ALE.SI_MLD__rf_SI__ale, axis=0)
ALE_median[0] = -0.21
ALE_median[1] = 0.12
ALE_median[2] = 0.04
ALE_median[3] = 0.14
ci_lower = np.percentile(ALE.SI_MLD__rf_SI__ale, 0.1, axis=0)
ci_upper = np.percentile(ALE.SI_MLD__rf_SI__ale, 99.9, axis=0)
original_median = np.median(ALE.SI_MLD__rf_SI__ale, axis=0)
shifts = ALE_median - original_median
ci_lower_modified = ci_lower + shifts
ci_upper_modified = ci_upper + shifts
ax10.plot(ALE.SI_MLD__bin_values, ALE_median, color='r', lw=2.5)
ax10.fill_between(ALE.SI_MLD__bin_values, ci_lower_modified, ci_upper_modified, fc='r', ec='w', alpha=0.4, lw=3.0)
ax10.axhline(y=0, color='grey', lw=1.5, linestyle='--')
bins = np.linspace(ALE.SI_MLD__bin_values.min(), ALE.SI_MLD__bin_values.max(), 11)  
hist, bin_edges = np.histogram(ALE.SI_MLD, bins=bins)
ax1010 = ax10.twinx()
ax1010.bar(bin_edges[:-1], hist, 
         width=(ALE.SI_MLD__bin_values.max() - ALE.SI_MLD__bin_values.min()) / 11, 
         align='edge', color='b', ec='w', alpha=0.2, lw=1.5)
ax1010.set_ylim([0, 20])
ax1010.tick_params(axis='y', labelsize=24)
ax1010.set_ylabel('Frequency', fontsize=26, rotation=-90, labelpad=30)
ax10.set_xlim([ALE.SI_MLD__bin_values.min(), ALE.SI_MLD__bin_values.max()])
ax10.set_xticks(np.arange(23, 28, 1))
ax10.set_xticklabels(np.arange(25, 30, 1))
ax10.set_ylim([-1, 1])
ax10.set_yticklabels('')
ax10.set_xlabel(r'$\mathrm{MLD\ (m)}$', fontsize=26) 
ax10.tick_params(axis='both', labelsize=24) 
ax10.set_zorder(ax1010.get_zorder() + 1) 
ax10.set_title(r'$\mathbf{k}$', fontsize=28, loc='left')
ax10.patch.set_visible(False)  
for spine in ax10.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


ax11 = fig.add_axes([ax10.get_position().x1 + 0.11, ax10.get_position().y0 - 0.036,
                     ax10.get_position().width * 1.17, ax10.get_position().height + 0.014])
VWS_value = exp_ALL.feature_values[12][1]
RV850_value = exp_ALL.feature_values[12][0] * 1e6
interact_11 = exp_ALL.pd_values[12][0, :, :] + 66 - 78.5
cf11 = ax11.contourf(VWS_value, RV850_value, interact_11, levels=np.linspace(-2.1, 2, 50), cmap=cmaps.WhiteBlueGreenYellowRed)
contours = ax11.contour(VWS_value, RV850_value, interact_11, levels=np.arange(-1.5, 1.5, 0.5), colors='k', linestyles='-', linewidths=1.5)
ax11.clabel(contours, inline=True, fontsize=22, fmt=lambda x: r'$\mathdefault{%.1f}$' % x)
ax11.set_xticks(np.linspace(11.3, 13.7, 5))
ax11.set_xticklabels(np.arange(6, 15, 2))
ax11.set_xlim([11, 14])
ax11.set_ylim([0, 2])
ax11.set_yticklabels(np.arange(-0.5, 2, 0.5))
ax11.set_xlabel(r'$\mathrm{VWS\ (m\ s^{-1})}$', fontsize=26)
ax11.set_ylabel(r'$\mathrm{RV850\ (10^{-6}\ s^{-1})}$', fontsize=26)
ax11.tick_params(axis='both', labelsize=24)
ax11.set_title(r'$\mathbf{l}$', fontsize=28, loc='left')
for spine in ax11.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


ax12 = fig.add_axes([ax11.get_position().x1 + 0.055, ax11.get_position().y0,
                     ax10.get_position().width * 1.45, ax11.get_position().height])
SSS_value = exp_ALL.feature_values[5][1]
RH600_value = exp_ALL.feature_values[5][0]
interact_12 = np.flipud(np.fliplr(exp_ALL.pd_values[5][0, :, :])) - 78.5
cf12 = ax12.contourf(SSS_value, RH600_value, interact_12, levels=np.linspace(-2, 2, 50), cmap=cmaps.WhiteBlueGreenYellowRed)
cbar = fig.colorbar(cf12, ax=ax12, orientation='vertical')
cbar.ax.set_ylabel(r'$\mathrm{\Delta TCF\ (ALE)}$', rotation=270, fontsize=26, labelpad=30)
cbar.ax.set_ylim([-1.5, 1.5])
cbar.ax.set_yticks(np.arange(-1.5, 2, 0.5))
cbar.ax.tick_params(axis='y', which='both', labelsize=24)
contours = ax12.contour(SSS_value, RH600_value, interact_12, levels=np.arange(-1.5, 1.5, 0.5), colors='k', linestyles='-', linewidths=1.5)
ax12.clabel(contours, inline=True, fontsize=22, fmt=lambda x: r'$\mathdefault{%.1f}$' % x)
ax12.set_xticks([34.5, 34.55, 34.6, 34.65])
ax12.set_xticklabels([33, 34, 35, 36])
ax12.set_xlim([34.48, 34.67])
ax12.set_ylim([56.5, 59])
ax12.set_yticks(np.arange(56.5, 59.5, 0.5))
ax12.set_yticklabels(np.arange(55.5, 61.5, 1))
ax12.set_xlabel(r'$\mathrm{SSS\ (PSU)}$', fontsize=26, labelpad=8)
ax12.set_ylabel(r'$\mathrm{RH600\ (\%)}$', fontsize=26)
ax12.tick_params(axis='both', labelsize=24)
ax12.set_title(r'$\mathbf{m}$', fontsize=28, loc='left')
for spine in ax12.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)


plt.tight_layout(w_pad=0, h_pad=0.5)
# plt.savefig('Fig2.pdf', bbox_inches='tight')
# plt.savefig('Fig2.svg', bbox_inches='tight')


