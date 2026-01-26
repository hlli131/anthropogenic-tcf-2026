import cmaps
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


# plot
fig = plt.figure(figsize=(15, 12), dpi=500)
gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1]) 

ax1 = plt.subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax1.add_feature(cfeature.LAND, fc='w')
cmap = cm.RdBu_r
cmap = ListedColormap(cmap(np.linspace(0, 1, 12)))
norm = BoundaryNorm(boundaries=np.linspace(-0.15, 0.15, 13), ncolors=12, extend='neither')
pmesh = ax1.pcolormesh(Annual_TCF_2p5.lon, Annual_TCF_2p5.lat, trend, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
lons, lats = np.meshgrid(Annual_TCF_2p5.lon, Annual_TCF_2p5.lat)
mask = p_value < 0.05
ax1.scatter(lons[mask], lats[mask], marker='o', s=6, c='k', transform=ccrs.PlateCarree())
cbar = fig.colorbar(pmesh, ax=ax1, orientation='horizontal', pad=0.07, fraction=0.1, aspect=50)
cbar.ax.set_xlabel(r'$\mathrm{Annual\ TCF\ trend\ (decade^{-1})}$', fontsize=20, labelpad=10)
cbar.ax.grid(True, which='both', axis='both', lw=1.5, linestyle='-', c='k')
cbar.ax.tick_params(axis='x', which='both', length=0, width=0)
MDRs = [
    {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20},  
]
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=2, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax1.add_patch(rectangle)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(1.5)
ax1.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax1.set_aspect('equal')
ax1.set_title(r'$\mathbf{a}$', fontsize=22, loc='left')
ax1.set_xticks(np.arange(-180, 181, 60))
ax1.set_yticks(np.arange(-50, 51, 25))
ax1.set_xticks(np.arange(-180, 181, 20), minor=True)
ax1.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax1.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax1.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out', labelsize=18)
ax1.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out', labelsize=18)
ax1.xaxis.set_tick_params(which='minor', length=6, width=1.5, color='k', direction='out', labelsize=18)
ax1.yaxis.set_tick_params(which='minor', length=6, width=1.5, color='k', direction='out', labelsize=18)
ax1.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
for spine in ax1.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')


ax2 = plt.subplot(gs[1, 0])
region = 'All'
ax2.plot(anomaly['SEASON'], anomaly[region], color='darkblue', lw=1.5, alpha=0.5)
ax2.plot(anomaly['SEASON'], intercept + slope * anomaly['SEASON'], linestyle='--', color='darkblue', lw=2)
p_text = f"$p<0.0015$" if p < 0.001 else f"$p={p:.3f}$"
ax2.text(0.97, 0.03, f'$\\mathrm{{Trend={slope:.2f}}}$\n{p_text}', fontsize=18, color='k', transform=ax2.transAxes, ha='right', va='bottom')
smoothed_data = anomaly[region].rolling(window=5, center=True).mean()
ax2.plot(anomaly['SEASON'], smoothed_data, linestyle='-', color='darkblue', lw=1.5)
ax2.axhline(y=0, color='k', alpha=0.5, lw=1.5, linestyle='--')
ax2.set_ylim([-20, 20])
ax2.set_xlim([1980, 2020])
ax2.tick_params(axis='both', labelsize=18)
ax2.set_ylabel('TCF anomaly', fontsize=18)
ax2.set_title(r'$\mathbf{b}$', fontsize=22, loc='left')
ax2.set_title('Global', fontsize=22)
ax2.xaxis.set_minor_locator(MultipleLocator(5))
ax2.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=6)
ax2.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=6)
ax2.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax2.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax2.spines.values():
        spine.set_linewidth(1.5)


ax3 = plt.subplot(gs[1, 1])
region = 'NH'
ax3.plot(anomaly['SEASON'], anomaly[region], color='g', lw=1.5, alpha=0.5)
ax3.plot(anomaly['SEASON'], intercept + slope * anomaly['SEASON'], linestyle='--', color='g', lw=2)
p_text = f"$p<0.0015$" if p < 0.001 else f"$p={p:.3f}$"
ax3.text(0.97, 0.03, f'$\\mathrm{{Trend={slope:.2f}}}$\n{p_text}', fontsize=18, color='k', transform=ax3.transAxes, ha='right', va='bottom')
smoothed_data = anomaly[region].rolling(window=5, center=True).mean()
ax3.plot(anomaly['SEASON'], smoothed_data, linestyle='-', color='g', lw=1.5)
ax3.axhline(y=0, color='k', alpha=0.5, lw=1.5, linestyle='--')
ax3.set_ylim([-20, 20])
ax3.set_xlim([1980, 2020])
ax3.tick_params(axis='both', labelsize=18)
ax3.set_title(r'$\mathbf{c}$', fontsize=22, loc='left')
ax3.set_title('NH', fontsize=22)
ax3.xaxis.set_minor_locator(MultipleLocator(5))
ax3.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=6)
ax3.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=6)
ax3.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax3.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax3.spines.values():
        spine.set_linewidth(1.5)


ax4 = plt.subplot(gs[1, 2])
ax4.plot(anomaly['SEASON'], anomaly[region], color='purple', lw=1.5, alpha=0.5)
ax4.plot(anomaly['SEASON'], intercept + slope * anomaly['SEASON'], linestyle='--', color='purple', lw=2)
p_text = f"$p<0.0015$" if p < 0.001 else f"$p={p:.3f}$"
ax4.text(0.97, 0.03, f'$\\mathrm{{Trend={slope:.2f}}}$\n{p_text}', fontsize=18, color='k', transform=ax4.transAxes, ha='right', va='bottom')
smoothed_data = anomaly[region].rolling(window=5, center=True).mean()
ax4.plot(anomaly['SEASON'], smoothed_data, linestyle='-', color='purple', lw=1.5)
ax4.axhline(y=0, color='k', alpha=0.5, lw=1.5, linestyle='--')
ax4.set_ylim([-10, 10])
ax4.set_xlim([1980, 2020])
ax4.tick_params(axis='both', labelsize=18)
ax4.set_title(r'$\mathbf{d}$', fontsize=22, loc='left')
ax4.set_title('SH', fontsize=22)
ax4.xaxis.set_minor_locator(MultipleLocator(5))
ax4.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=6)
ax4.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=6)
ax4.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax4.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax4.spines.values():
        spine.set_linewidth(1.5)

plt.tight_layout(w_pad=1, h_pad=0.5)
# plt.savefig('Fig1.pdf', bbox_inches='tight')
# plt.savefig('Fig1.svg', bbox_inches='tight')