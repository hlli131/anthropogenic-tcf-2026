import xesmf as xe
import numpy as np
import xarray as xr
from xMCA import xMCA
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, spearmanr
from eofs.xarray import Eof
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'  
mpl.rcParams['font.size'] = 18


MDRs = [
    {'name': 'WNP', 'lon_min': 120, 'lon_max': 160, 'lat_min': 5, 'lat_max': 25},
    {'name': 'ENP', 'lon_min': 240, 'lon_max': 270, 'lat_min': 5, 'lat_max': 20},
    {'name': 'NA', 'lon_min': 310, 'lon_max': 345, 'lat_min': 5, 'lat_max': 20},
    {'name': 'SI', 'lon_min': 55, 'lon_max': 105, 'lat_min': -15, 'lat_max': -5},
    {'name': 'SP', 'lon_min': 150, 'lon_max': 190, 'lat_min': -20, 'lat_max': -5},
    {'name': 'NI', 'lon_min': 60, 'lon_max': 95, 'lat_min': 5, 'lat_max': 20},  
]

fig = plt.figure(figsize=(18, 12), dpi=500)
gs = gridspec.GridSpec(3, 2) 

ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax1.add_feature(cfeature.LAND, fc='w')
pmesh = ax1.pcolormesh(lp[0].lon, lp[0].lat, -lp[0].T.where(lp[0].T != 0), transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.1, vmax=0.1) 
ax1.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax1.set_aspect('equal')
ax1.set_xticks(np.arange(-180, 181, 60))
ax1.set_yticks(np.arange(-50, 51, 25))
ax1.set_xticks(np.arange(-180, 181, 20), minor=True)
ax1.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax1.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax1.tick_params(axis='both', labelsize=18)
ax1.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax1.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax1.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax1.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax1.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax1.set_title(r'$\mathbf{a}$', fontsize=22, loc='left')
ax1.set_title('67%', fontsize=20, loc='right')
ax1.set_title('SVD1 TCF', fontsize=22)
for spine in ax1.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=2, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax1.add_patch(rectangle)


ax2 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax2.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax2.add_feature(cfeature.LAND, fc='w')
pmesh2 = ax2.pcolormesh(rp[1].lon, rp[1].lat, -rp[1], transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.08, vmax=0.08) 
cbar2 = fig.colorbar(pmesh2, ax=ax2, orientation='horizontal', pad=0.2, fraction=0.08, aspect=40, extend='both')
cbar2.ax.tick_params(axis='x', which='both', length=5, width=1.5)
cbar2.outline.set_linewidth(1.5)
ax2.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax2.set_aspect('equal')
ax2.set_xticks(np.arange(-180, 181, 60))
ax2.set_yticks(np.arange(-50, 51, 25))
ax2.set_xticks(np.arange(-180, 181, 20), minor=True)
ax2.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax2.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax2.tick_params(axis='both', labelsize=18)
ax2.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax2.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax2.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax2.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax2.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax2.set_title(r'$\mathbf{b}$', fontsize=22, loc='left')
ax2.set_title('67%', fontsize=20, loc='right')
ax2.set_title('SVD1 SST', fontsize=22)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')


ax3 = plt.subplot(gs[2, 0])
ax3.plot(np.arange(1980, 2021), -le[1], linestyle='-', color='#4DBBD5FF', lw=2, label=r'$\mathrm{EC1_{TCF}}$')
ax3.plot(np.arange(1980, 2021), -re[1], linestyle='-', color='#E64B35FF', lw=2, label=r'$\mathrm{EC1_{SST}}$')
ax3.plot(np.arange(1980, 2021), sst_zscore, linestyle='-', color='#3C5488FF', lw=2, label='Global mean SST')
ax3.legend(frameon=False, fontsize=18)
ax3.axhline(y=0, color='k', alpha=0.5, lw=1.5, linestyle='--')
ax3.text(0.98, 0.05, f'$r=0.85, P<0.001$', fontsize=18, color='#3C5488FF', transform=ax3.transAxes, ha='right', va='bottom')
ax3.set_xlim([1980, 2020])
ax3.set_ylim([-3, 3])
ax3.tick_params(axis='both', labelsize=18)
ax3.set_title(r'$\mathbf{c}$', fontsize=22, loc='left')
ax3.set_title('EC1 & Global mean SST', fontsize=22)
ax3.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=6)
ax3.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=6)
ax3.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax3.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax3.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')


ax4 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=180))
ax4.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax4.add_feature(cfeature.LAND, fc='w')
pmesh = ax4.pcolormesh(lp[1].lon, lp[1].lat, -lp[1].T.where(lp[1].T != 0), transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.1, vmax=0.1) 
ax4.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax4.set_aspect('equal')
ax4.set_xticks(np.arange(-180, 181, 60))
ax4.set_yticks(np.arange(-50, 51, 25))
ax4.set_xticks(np.arange(-180, 181, 20), minor=True)
ax4.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax4.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax4.tick_params(axis='both', labelsize=18)
ax4.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax4.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax4.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax4.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax4.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax4.set_title(r'$\mathbf{d}$', fontsize=22, loc='left')
ax4.set_title('24%', fontsize=20, loc='right')
ax4.set_title('SVD2 TCF', fontsize=22)
for spine in ax4.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')
for MDR in MDRs:
    rectangle = patches.Rectangle((MDR['lon_min'], MDR['lat_min']), MDR['lon_max'] - MDR['lon_min'], MDR['lat_max'] - MDR['lat_min'], 
                                   ls='-', lw=2, ec='r', fc='none', transform=ccrs.PlateCarree())
    ax4.add_patch(rectangle)


ax5 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree(central_longitude=180))
ax5.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax5.add_feature(cfeature.LAND, fc='w')
pmesh5 = ax5.pcolormesh(rp[0].lon, rp[0].lat, -rp[0], transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.08, vmax=0.08)
cbar5 = fig.colorbar(pmesh5, ax=ax5, orientation='horizontal', pad=0.2, fraction=0.08, aspect=40, extend='both') 
cbar5.ax.tick_params(axis='x', which='both', length=5, width=1.5)
cbar5.outline.set_linewidth(1.5)
ax5.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax5.set_aspect('equal')
ax5.set_xticks(np.arange(-180, 181, 60))
ax5.set_yticks(np.arange(-50, 51, 25))
ax5.set_xticks(np.arange(-180, 181, 20), minor=True)
ax5.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax5.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax5.tick_params(axis='both', labelsize=18)
ax5.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax5.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax5.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax5.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax5.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax5.set_title(r'$\mathbf{e}$', fontsize=22, loc='left')
ax5.set_title('24%', fontsize=20, loc='right')
ax5.set_title('SVD2 SST', fontsize=22)
for spine in ax5.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')


ax6 = plt.subplot(gs[2, 1])
ax6.plot(np.arange(1980, 2021), -le[0], linestyle='-', color='#4DBBD5FF', lw=2, label=r'$\mathrm{EC2_{TCF}}$')
ax6.plot(np.arange(1980, 2021), -re[0], linestyle='-', color='#E64B35FF', lw=2, label=r'$\mathrm{EC2_{SST}}$')
ax6.plot(np.arange(1980, 2021), amo_zscore, linestyle='-', color='#12AC1AFF', lw=2, label='AMO index')
ax6.plot(np.arange(1980, 2021), -ipo, linestyle='-', color='purple', lw=2, label='$-$IPO index')
ax6.legend(frameon=False, fontsize=18, ncol=2)
ax6.axhline(y=0, color='k', alpha=0.5, lw=1.5, linestyle='--')
ax6.text(0.98, 0.18, f'$r=0.68, P<0.001$', fontsize=18, color='#12AC1AFF', transform=ax6.transAxes, ha='right', va='bottom')
ax6.text(0.98, 0.05, f'$r=0.61, P<0.001$', fontsize=18, color='purple', transform=ax6.transAxes, ha='right', va='bottom')
ax6.set_xlim([1980, 2020])
ax6.set_ylim([-4, 4])
ax6.tick_params(axis='both', labelsize=18)
ax6.set_title(r'$\mathbf{f}$', fontsize=22, loc='left')
ax6.set_title('EC2 & Climate index', fontsize=22)
ax6.xaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=6)
ax6.xaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=6)
ax6.yaxis.set_tick_params(which='major', length=6, width=1.5, color='k', direction='out', pad=4)
ax6.yaxis.set_tick_params(which='minor', length=4, width=1.5, color='k', direction='out', pad=4)
for spine in ax6.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k')


plt.tight_layout(w_pad=1)
row1_bottom = gs[0, 0].get_position(fig).y0
new_row1_bottom = row1_bottom - 0.002
for ax in [ax1, ax4]: 
    pos = ax.get_position()
    ax.set_position([pos.x0, new_row1_bottom, pos.width, pos.height])

# plt.savefig('Fig3.pdf', bbox_inches='tight')
# plt.savefig('Fig3.svg', bbox_inches='tight')