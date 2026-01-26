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



# conduct detection and attribution analysis
MDRs = [
    {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20},  
]

fig = plt.figure(figsize=(20, 9), dpi=300)
gs = gridspec.GridSpec(3, 3, width_ratios=[5, 3.2, 1.8]) 
ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax2 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax3 = plt.subplot(gs[2, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax4 = plt.subplot(gs[0, 1])
ax5 = plt.subplot(gs[1, 1])
ax6 = plt.subplot(gs[2, 1])
ax7 = plt.subplot(gs[0, 2])
ax8 = plt.subplot(gs[1, 2])
ax9 = plt.subplot(gs[2, 2])


ax1.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax1.add_feature(cfeature.LAND, fc='w')
cmap = cm.RdBu_r
cmap = ListedColormap(cmap(np.linspace(0, 1, 12)))
norm = BoundaryNorm(boundaries=np.linspace(-0.09, 0.09, 13), ncolors=12, extend='neither')
pmesh = ax1.pcolormesh(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat, trend_ALL transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
lons, lats = np.meshgrid(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat)
mask = p_ALL < 0.05
ax1.scatter(lons[mask], lats[mask], c='k', marker='o', s=1, transform=ccrs.PlateCarree())
cbar = fig.colorbar(pmesh, ax=ax1, orientation='vertical', pad=0.02,  aspect=13)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((0, 0)) 
cbar.ax.yaxis.get_offset_text().set_fontsize(16)
cbar.ax.set_ylabel(r'$\mathrm{decade^{-1}}$', rotation=270, fontsize=20, labelpad=25)
cbar.ax.grid(True, which='both', axis='both', lw=1.5, linestyle='-', c='k')
cbar.ax.tick_params(axis='y', which='both', length=0, width=0, labelsize=18)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(1.5)
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=1.5, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax1.add_patch(rectangle)
ax1.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax1.set_aspect('equal')
ax1.set_title(r'$\mathbf{a}$', fontsize=22, loc='left')
ax1.set_title('ALL simulation', fontsize=22)
ax1.set_xticks(np.arange(-180, 181, 60))
ax1.set_yticks(np.arange(-50, 51, 25))
ax1.set_xticks(np.arange(-180, 181, 20), minor=True)
ax1.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'], fontsize=18)
ax1.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'], fontsize=18)
ax1.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax1.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax1.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax1.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax1.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
for spine in ax1.spines.values():
    spine.set_linewidth(1.5) 


ax2.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax2.add_feature(cfeature.LAND, fc='w')
cmap = cm.RdBu_r
cmap = ListedColormap(cmap(np.linspace(0, 1, 12)))
norm = BoundaryNorm(boundaries=np.linspace(-0.09, 0.09, 13), ncolors=12, extend='neither')
pmesh = ax2.pcolormesh(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat, trend_GHG, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
lons, lats = np.meshgrid(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat)
mask = p_GHG < 0.05
ax2.scatter(lons[mask], lats[mask], c='k', marker='o', s=1, transform=ccrs.PlateCarree())
cbar = fig.colorbar(pmesh, ax=ax2, orientation='vertical', pad=0.02,  aspect=13)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((0, 0)) 
cbar.ax.yaxis.get_offset_text().set_fontsize(16)
cbar.ax.set_ylabel(r'$\mathrm{decade^{-1}}$', rotation=270, fontsize=20, labelpad=25)
cbar.ax.grid(True, which='both', axis='both', lw=1.5, linestyle='-', c='k')
cbar.ax.tick_params(axis='y', which='both', length=0, width=0, labelsize=18)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(1.5)
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=1.5, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax2.add_patch(rectangle)
ax2.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax2.set_aspect('equal')
ax2.set_title(r'$\mathbf{b}$', fontsize=22, loc='left')
ax2.set_title('GHG simulation', fontsize=22)
ax2.set_xticks(np.arange(-180, 181, 60))
ax2.set_yticks(np.arange(-50, 51, 25))
ax2.set_xticks(np.arange(-180, 181, 20), minor=True)
ax2.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'], fontsize=18)
ax2.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'], fontsize=18)
ax2.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax2.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax2.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax2.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax2.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
for spine in ax2.spines.values():
    spine.set_linewidth(1.5) 


ax3.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax3.add_feature(cfeature.LAND, fc='w')
cmap = cm.RdBu_r
cmap = ListedColormap(cmap(np.linspace(0, 1, 12)))
norm = BoundaryNorm(boundaries=np.linspace(-0.09, 0.09, 13), ncolors=12, extend='neither')
pmesh = ax3.pcolormesh(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat, trend_AER, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
lons, lats = np.meshgrid(Annual_ENGPI_2p5.lon, Annual_ENGPI_2p5.lat)
mask = p_GHG < 0.05
ax3.scatter(lons[mask], lats[mask], c='k', marker='o', s=1, transform=ccrs.PlateCarree())
cbar = fig.colorbar(pmesh, ax=ax3, orientation='vertical', pad=0.02,  aspect=13)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((0, 0)) 
cbar.ax.yaxis.get_offset_text().set_fontsize(16)
cbar.ax.set_ylabel(r'$\mathrm{decade^{-1}}$', rotation=270, fontsize=20, labelpad=25)
cbar.ax.grid(True, which='both', axis='both', lw=1.5, linestyle='-', c='k')
cbar.ax.tick_params(axis='y', which='both', length=0, width=0, labelsize=18)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(1.5)
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=1.5, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax3.add_patch(rectangle)
ax3.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax3.set_aspect('equal')
ax3.set_title(r'$\mathbf{c}$', fontsize=22, loc='left')
ax3.set_title('AER simulation', fontsize=22)
ax3.set_xticks(np.arange(-180, 181, 60))
ax3.set_yticks(np.arange(-50, 51, 25))
ax3.set_xticks(np.arange(-180, 181, 20), minor=True)
ax3.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'], fontsize=18)
ax3.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'], fontsize=18)
ax3.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax3.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax3.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax3.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax3.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
for spine in ax3.spines.values():
    spine.set_linewidth(1.5) 


width=0.35
colors = ['#E64B35FF', '#14AD96FF', '#3C5488FF', '#EC9E27', '#979595',
          '#14AD96FF', '#3C5488FF', '#14AD96FF', '#EC9E27', '#979595']
bars_array = np.array([
    # index, lower, upper, mean
    [0, 0.5, 2, 1.4],      # ALL
    [1, -4, 3.8, 0.4],     # NAT
    [2, 0.7, 2.8, 1.8],    # ANT
    [3, -2.2, -5.5, -4],   # GHG
    [4, 1.5, 4.7, 3.1],    # AER
    [6, -5.4, 2.7, -1.1],  # NAT
    [7, 0.5, 3.6, 2.1],    # ANT
    [9, -0.9, 3.4, 1.5],   # NAT
    [10, -6.5, 0.7, -3.4], # GHG
    [11, 0.4, 3.6, 2.1]    # AER
])
for (x_pos, bottom, top, hline_y), color in zip(bars_array, colors):
    height = top - bottom
    if height < 0:
        bottom, top = top, bottom
        height = -height
    ax4.bar(x_pos, height, width=width, align='center', color=color, bottom=bottom)
    ax4.hlines(hline_y, x_pos - width / 2, x_pos + width / 2, colors='w', linestyles='-', lw=2.5)
ax4.axvline(x=5, color='k', lw=1.5)
ax4.axvline(x=8, color='k', lw=1.5)
ax4.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax4.axhline(y=1, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax4.text(2.25, 4.8, '1-signal', ha='center', va='center', fontsize=18)
ax4.text(6.5, 4.8, '2-signal', ha='center', va='center', fontsize=18)
ax4.text(9.75, 4.8, '3-signal', ha='center', va='center', fontsize=18)
ax4.set_xlim([-0.5, 11.5])
ax4.set_xticks([0, 1, 2, 3, 4, 6, 7, 9, 10, 11])
ax4.set_ylim([-6, 6])
ax4.set_yticks([-6, -3, 0, 3, 6])
ax4.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER', 'NAT', 'ANT', 'NAT', 'GHG', 'AER'], rotation=45)
ax4.set_ylabel('Scaling factor', fontsize=20)
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=18)
ax4.set_title(r'$\mathbf{d}$', fontsize=22, loc='left')
ax4.set_title('Global', fontsize=22)
ax4.yaxis.set_minor_locator(MultipleLocator(1))
ax4.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax4.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax4.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax4.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax4.spines.values():
        spine.set_linewidth(1.5)


bars_array = np.array([
  # index, lower, upper, mean
    [0, 1.5, 4.7, 3],      # ALL
    [1, -4.2, 3.5, -0.6],  # NAT
    [2, 0.3, 2.4, 1.4],    # ANT
    [3, -1.9, -6.3, -4.9], # GHG
    [4, 0.6, 4.2, 2.6],    # AER
    [6, -8, 1.5, -2.8],    # NAT
    [7, 3, 4.8, 3.8],      # ANT
    [9, -1.5, 4.8, 2.1],   # NAT
    [10, -6, -1.8, -3.6],  # GHG
    [11, 1.9, 4.9, 3.5]    # AER
])
for (x_pos, bottom, top, hline_y), color in zip(bars_array, colors):
    height = top - bottom
    if height < 0:
        bottom, top = top, bottom
        height = -height
    ax5.bar(x_pos, height, width=width, align='center', color=color, bottom=bottom)
    ax5.hlines(hline_y, x_pos - width / 2, x_pos + width / 2, colors='w', linestyles='-', lw=2.5)
ax5.axvline(x=5, color='k', lw=1.5)
ax5.axvline(x=8, color='k', lw=1.5)
ax5.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax5.axhline(y=1, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax5.set_xlim([-0.6, 11.6])
ax5.set_xticks([0, 1, 2, 3, 4, 6, 7, 9, 10, 11])
ax5.set_ylim([-6, 6])
ax5.set_yticks([-6, -3, 0, 3, 6])
ax5.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER', 'NAT', 'ANT', 'NAT', 'GHG', 'AER'], rotation=45)
ax5.set_ylabel('Scaling factor', fontsize=20)
ax5.tick_params(axis='x', labelsize=16)
ax5.tick_params(axis='y', labelsize=18)
ax5.set_title(r'$\mathbf{e}$', fontsize=22, loc='left')
ax5.set_title('NH', fontsize=22)
ax5.yaxis.set_minor_locator(MultipleLocator(1))
ax5.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax5.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax5.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax5.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax5.spines.values():
        spine.set_linewidth(1.5)


bars_array = np.array([
    [0, 0.5, 3.2, 1.7],   
    [1, -3.1, 5.2, 1.5],  
    [2, 0.7, 3.6, 2.2],
    [3, 2.9, 4.3, 3.7],  
    [4, -6, -2, -4.5],
    [6, -3.9, 2.4, -0.7],
    [7, 1.9, 4.4, 3.2],
    [9, -1.7, 2.2, 0.5],
    [10, 1.9, 5.1, 3.7],
    [11, -5.2, 0.6, -2.5]
])
for (x_pos, bottom, top, hline_y), color in zip(bars_array, colors):
    height = top - bottom
    if height < 0:
        bottom, top = top, bottom
        height = -height
    ax6.bar(x_pos, height, width=width, align='center', color=color, bottom=bottom)
    ax6.hlines(hline_y, x_pos - width / 2, x_pos + width / 2, colors='w', linestyles='-', lw=2.5)
ax6.axvline(x=5, color='k', lw=1.5)
ax6.axvline(x=8, color='k', lw=1.5)
ax6.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax6.axhline(y=1, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax6.set_xlim([-0.6, 11.6])
ax6.set_xticks([0, 1, 2, 3, 4, 6, 7, 9, 10, 11])
ax6.set_ylim([-6, 6])
ax6.set_yticks([-6, -3, 0, 3, 6])
ax6.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER', 'NAT', 'ANT', 'NAT', 'GHG', 'AER'], rotation=45)
ax6.set_ylabel('Scaling factor', fontsize=20)
ax6.tick_params(axis='x', labelsize=16)
ax6.tick_params(axis='y', labelsize=18)
ax6.set_title(r'$\mathbf{f}$', fontsize=22, loc='left')
ax6.set_title('SH', fontsize=22)
ax6.yaxis.set_minor_locator(MultipleLocator(1))
ax6.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out')
ax6.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out')
ax6.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax6.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax6.spines.values():
        spine.set_linewidth(1.5)


widths = 0.5
colors = ['#E64B35FF', '#14AD96FF', '#3C5488FF', '#EC9E27', '#979595']
stats = [
    # [med, q1, q3, lower_whisker, upper_whisker, mean]
    [5, 0.6, 1.7, -0.3, 2.2, 1.2],  # ALL
    [2, -0.8, 0.2, -1.4, 0.9, -0.3],  # NAT
    [4, 0.8, 1.8, -0.2, 2.7, 1.4],  # ANT
    [7, -2.1, -0.8, -3.2, 0.4, -1.5],  # GHG
    [3, 1.1, 2.9, -0.3, 3.6, 2.1]  # AER
]
boxplot_stats = []
for i, stat in enumerate(stats):
    boxplot_stats.append({
        'med': stat[0],    
        'q1': stat[1],   
        'q3': stat[2],    
        'whislo': stat[3], 
        'whishi': stat[4], 
        'mean': stat[5], 
        'fliers': [],     
        'label': ['ALL', 'NAT', 'ANT', 'GHG', 'AER'][i] 
    })
boxplots = ax7.bxp(boxplot_stats, widths=widths, showfliers=False, showmeans=True, meanline=True, patch_artist=True)
for patch, color in zip(boxplots['boxes'], colors):
    patch.set(fc=color, ec='none')
for cap in boxplots['caps']:
    cap.set_visible(False)
for i, whisker in enumerate(boxplots['whiskers']):
    whisker.set(color=colors[i // 2], lw=1.5)
for mean in boxplots['means']:
    mean.set(color='w', lw=2.5, linestyle='-')
for median in boxplots['medians']:
    median.set_color('none')
ax7.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax7.axhline(y=2.6, color='k', lw=1.5, zorder=10)
ax7.set_xlim([0.5, 5.5])
ax7.set_xticks([1, 2, 3, 4, 5])
ax7.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER'])
ax7.set_ylim([-4, 4])
ax7.set_yticks([-4, -2, 0, 2, 4])
ax7.set_ylabel('Attributable trend' + '\n', fontsize=20)
ax7.text(-0.15, 0.5, r'$\mathrm{(decade^{-1})}$', fontsize=19,  
         rotation=90, transform=ax7.transAxes, va='center', ha='right')
ax7.tick_params(axis='x', labelsize=16)
ax7.tick_params(axis='y', labelsize=18)
ax7.set_title(r'$\mathbf{g}$', fontsize=22, loc='left')
ax7.set_title('Global', fontsize=22)
ax7.xaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
ax7.yaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
for spine in ax7.spines.values():
    spine.set_linewidth(1.5)


stats = [
    # [med, q1, q3, lower_whisker, upper_whisker, mean]
    [5, 0.7, 2, -0.3, 2.9, 1.4],       # ALL
    [2, -1.2, 0.2, -2, 1.1, -0.5],     # NAT
    [4, 1.3, 2.3, 0.5, 3.4, 1.9],      # ANT
    [7, -2.8, -1, -3.6, -0.2, -1.9], # GHG
    [3, 1, 2.8, -0.3, 3.8, 1.8]        # AER
]
boxplot_stats = []
for i, stat in enumerate(stats):
    boxplot_stats.append({
        'med': stat[0],    
        'q1': stat[1],   
        'q3': stat[2],    
        'whislo': stat[3], 
        'whishi': stat[4], 
        'mean': stat[5], 
        'fliers': [],     
        'label': ['ALL', 'NAT', 'ANT', 'GHG', 'AER'][i] 
    })
boxplots = ax8.bxp(boxplot_stats, widths=widths, showfliers=False, showmeans=True, meanline=True, patch_artist=True)
for patch, color in zip(boxplots['boxes'], colors):
    patch.set(fc=color, ec='none')
for cap in boxplots['caps']:
    cap.set_visible(False)
for i, whisker in enumerate(boxplots['whiskers']):
    whisker.set(color=colors[i // 2], lw=1.5)
for mean in boxplots['means']:
    mean.set(color='w', lw=2.5, linestyle='-')
for median in boxplots['medians']:
    median.set_color('none')
ax8.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax8.axhline(y=3.1, color='k', lw=1.5, zorder=10)
ax8.set_xlim([0.5, 5.5])
ax8.set_xticks([1, 2, 3, 4, 5])
ax8.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER'])
ax8.set_ylim([-4, 4])
ax8.set_yticks([-4, -2, 0, 2, 4])
ax8.set_ylabel('Attributable trend' + '\n', fontsize=20)
ax8.text(-0.15, 0.5, r'$\mathrm{(decade^{-1})}$', fontsize=19,
         rotation=90, transform=ax8.transAxes, va='center', ha='right')
ax8.tick_params(axis='x', labelsize=16)
ax8.tick_params(axis='y', labelsize=18)
ax8.set_title(r'$\mathbf{h}$', fontsize=22, loc='left')
ax8.set_title('NH', fontsize=22)
ax8.xaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
ax8.yaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
for spine in ax8.spines.values():
    spine.set_linewidth(1.5)


stats = [
    # [med, q1, q3, lower_whisker, upper_whisker, mean]
    [2.5, -0.4, 0.2, -1, 0.65, -0.15],     # ALL
    [1, -0.1, 0.45, -0.45, 0.85, 0.2],  # NAT
    [2.05, -0.65, -0.10, -0.95, 0.4, -0.35],  # ANT
    [3.7, -1.2, -0.3, -1.65, 0.4, -0.75],  # GHG
    [0.8, -0.4, 0.35, -0.95, 0.7, -0.1]    # AER
]
boxplot_stats = []
for i, stat in enumerate(stats):
    boxplot_stats.append({
        'med': stat[0],    
        'q1': stat[1],   
        'q3': stat[2],    
        'whislo': stat[3], 
        'whishi': stat[4], 
        'mean': stat[5], 
        'fliers': [],     
        'label': ['ALL', 'NAT', 'ANT', 'GHG', 'AER'][i] 
    })
boxplots = ax9.bxp(boxplot_stats, widths=widths, showfliers=False, showmeans=True, meanline=True, patch_artist=True)
for patch, color in zip(boxplots['boxes'], colors):
    patch.set(fc=color, ec='none')
for cap in boxplots['caps']:
    cap.set_visible(False)
for i, whisker in enumerate(boxplots['whiskers']):
    whisker.set(color=colors[i // 2], lw=1.5)
for mean in boxplots['means']:
    mean.set(color='w', lw=2.5, linestyle='-')
for median in boxplots['medians']:
    median.set_color('none')
ax9.axhline(y=0, color='k', alpha=0.4, lw=1.5, linestyle='--')
ax9.axhline(y=-0.6, color='k', lw=1.5, zorder=10)
ax9.set_xlim([0.5, 5.5])
ax9.set_xticks([1, 2, 3, 4, 5])
ax9.set_xticklabels(['ALL', 'NAT', 'ANT', 'GHG', 'AER'])
ax9.set_ylim([-2, 2])
ax9.set_yticks([-2, -1, 0, 1, 2])
ax9.set_ylabel('Attributable trend' + '\n', fontsize=20)
ax9.text(-0.15, 0.5, r'$\mathrm{(decade^{-1})}$', fontsize=19,  
         rotation=90, transform=ax9.transAxes, va='center', ha='right')
ax9.tick_params(axis='x', labelsize=16)
ax9.tick_params(axis='y', labelsize=18)
ax9.set_title(r'$\mathbf{i}$', fontsize=22, loc='left')
ax9.set_title('SH', fontsize=22)
ax9.xaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
ax9.yaxis.set_tick_params(which='both', length=4, width=1.5, color='k', direction='out')
for spine in ax9.spines.values():
    spine.set_linewidth(1.5)


plt.tight_layout(w_pad=-0.5, h_pad=0.5)
col2_left = gs[0, 1].get_position(fig).x0
new_col2_left = col2_left - 0.02
for ax in [ax4, ax5, ax6]: 
    pos = ax.get_position()
    ax.set_position([new_col2_left, pos.y0, pos.width, pos.height])

# plt.savefig('Fig4.pdf', bbox_inches='tight')
# plt.savefig('Fig4.svg', bbox_inches='tight')