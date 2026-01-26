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



# plot TC environmental fields
fig = plt.figure(figsize=(20, 13), dpi=500)
ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax1.add_feature(cfeature.LAND, fc='w')
sat_nh = sat.where(sat['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
u850_nh = u850.where(u850['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
v850_nh = v850.where(v850['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
delta_sat_nh = sat_nh.sel(time=slice('2001', '2020')).mean('time') - sat_nh.sel(time=slice('1980', '2000')).mean('time')
delta_u850_nh = u850_nh.sel(time=slice('2001', '2020')).mean('time') - u850_nh.sel(time=slice('1980', '2000')).mean('time')
delta_v850_nh = v850_nh.sel(time=slice('2001', '2020')).mean('time') - v850_nh.sel(time=slice('1980', '2000')).mean('time')
cf1 = ax1.contourf(delta_sat_nh.lon, delta_sat_nh.lat, delta_sat_nh, levels=np.linspace(-1.5, 1.5, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.BlueWhiteOrangeRed, extend='both')
cbar1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.15, aspect=50)
cbar1.ax.set_xlabel(r'$\mathrm{\Delta SAT\ (\degree C)}$', fontsize=18, labelpad=5)
cbar1.ax.set_xticks(np.arange(-1.5, 1.8, 0.3))
cbar1.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar1.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 3                   
lon2d, lat2d = np.meshgrid(delta_u850_nh.lon, delta_u850_nh.lat)
u_skip = delta_u850_nh.values[::skip, ::skip]
v_skip = delta_v850_nh.values[::skip, ::skip]
lon_skip = lon2d[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
q1 = ax1.quiver(lon_skip, lat_skip, u_skip, v_skip, scale=3, scale_units='inches',
                width=0.0015, headwidth=5, headlength=7, pivot='middle', transform=ccrs.PlateCarree())
qk1 = ax1.quiverkey(q1, 0.9, 1.05, 1, r'$\mathrm{1\ m\ s^{-1}}$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax1.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax1.set_aspect('equal')
ax1.set_xticks(np.arange(-180, 181, 60))
ax1.set_yticks(np.arange(-50, 51, 25))
ax1.set_xticks(np.arange(-180, 181, 20), minor=True)
ax1.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax1.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax1.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax1.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax1.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax1.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax1.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax1.set_title(r'$\mathbf{a}$', fontsize=22, loc='left')
ax1.set_title('SAT & 850-hPa wind (JJASON)', fontsize=22)
for spine in ax1.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 
    

ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
ax2.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax2.add_feature(cfeature.LAND, fc='w')
sat_sh = sat.where(sat['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
u850_sh = u850.where(u850['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
v850_sh = v850.where(v850['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
delta_sat_sh = sat_sh.sel(time=slice('2001', '2020')).mean('time') - sat_sh.sel(time=slice('1980', '2000')).mean('time')
delta_u850_sh = u850_sh.sel(time=slice('2001', '2020')).mean('time') - u850_sh.sel(time=slice('1980', '2000')).mean('time')
delta_v850_sh = v850_sh.sel(time=slice('2001', '2020')).mean('time') - v850_sh.sel(time=slice('1980', '2000')).mean('time')
cf2 = ax2.contourf(delta_sat_sh.lon, delta_sat_sh.lat, delta_sat_sh, levels=np.linspace(-1.5, 1.5, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.BlueWhiteOrangeRed, extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.15, aspect=50)
cbar2.ax.set_xlabel(r'$\mathrm{\Delta SAT\ (\degree C)}$', fontsize=18, labelpad=5)
cbar2.ax.set_xticks(np.arange(-1.5, 1.8, 0.3))
cbar2.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar2.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 3                   
lon2d, lat2d = np.meshgrid(delta_u850_sh.lon, delta_u850_sh.lat)
u_skip = delta_u850_sh.values[::skip, ::skip]
v_skip = delta_v850_sh.values[::skip, ::skip]
lon_skip = lon2d[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
q2 = ax2.quiver(lon_skip, lat_skip, u_skip, v_skip, scale=3, scale_units='inches',
                width=0.0015, headwidth=5, headlength=7, pivot='middle', transform=ccrs.PlateCarree())
qk2 = ax2.quiverkey(q2, 0.9, 1.05, 1, r'$\mathrm{1\ m\ s^{-1}}$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax2.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax2.set_aspect('equal')
ax2.set_xticks(np.arange(-180, 181, 60))
ax2.set_yticks(np.arange(-50, 51, 25))
ax2.set_xticks(np.arange(-180, 181, 20), minor=True)
ax2.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax2.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax2.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax2.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax2.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax2.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax2.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax2.set_title(r'$\mathbf{d}$', fontsize=22, loc='left')
ax2.set_title('SAT & 850-hPa wind (DJFMAM)', fontsize=22)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 


ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.PlateCarree(central_longitude=180))
ax3.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax3.add_feature(cfeature.LAND, fc='w')
vws_nh = vws.where(vws['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
u200_nh = u200.where(u200['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
v200_nh = v200.where(v200['time.month'].isin([6, 7, 8, 9, 10, 11]), drop=True)
delta_vws_nh = vws_nh.sel(time=slice('2001', '2020')).mean('time') - vws_nh.sel(time=slice('1980', '2000')).mean('time')
delta_u200_nh = u200_nh.sel(time=slice('2001', '2020')).mean('time') - u200_nh.sel(time=slice('1980', '2000')).mean('time')
delta_v200_nh = v200_nh.sel(time=slice('2001', '2020')).mean('time') - v200_nh.sel(time=slice('1980', '2000')).mean('time')
cf3 = ax3.contourf(delta_vws_nh.lon, delta_vws_nh.lat, delta_vws_nh, levels=np.linspace(-2, 2, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdBu_r, extend='both')
cbar3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', pad=0.15, aspect=50)
cbar3.ax.set_xlabel(r'$\mathrm{\Delta VWS\ (m\ s^{-1})}$', fontsize=18, labelpad=5)
cbar3.ax.set_xticks(np.arange(-2, 2.1, 0.4))
cbar3.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar3.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 3                  
lon2d, lat2d = np.meshgrid(delta_u200_nh.lon, delta_u200_nh.lat)
u_skip = delta_u200_nh.values[::skip, ::skip]
v_skip = delta_v200_nh.values[::skip, ::skip]
lon_skip = lon2d[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
q3 = ax3.quiver(lon_skip, lat_skip, u_skip, v_skip, scale=6, scale_units='inches',
                width=0.0015, headwidth=5, headlength=7, pivot='middle', transform=ccrs.PlateCarree())
qk3 = ax3.quiverkey(q3, 0.9, 1.05, 2, r'$\mathrm{2\ m\ s^{-1}}$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax3.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax3.set_aspect('equal')
ax3.set_xticks(np.arange(-180, 181, 60))
ax3.set_yticks(np.arange(-50, 51, 25))
ax3.set_xticks(np.arange(-180, 181, 20), minor=True)
ax3.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax3.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax3.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax3.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax3.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax3.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax3.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax3.set_title(r'$\mathbf{b}$', fontsize=22, loc='left')
ax3.set_title('VWS & 200-hPa wind (JJASON)', fontsize=22)
for spine in ax3.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 


ax4 = fig.add_subplot(3, 2, 4, projection=ccrs.PlateCarree(central_longitude=180))
ax4.add_feature(cfeature.COASTLINE, ec='k', lw=0.8)
ax4.add_feature(cfeature.LAND, fc='w')
vws_sh = vws.where(vws['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
u200_sh = u200.where(u200['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
v200_sh = v200.where(v200['time.month'].isin([12, 1, 2, 3, 4, 5]), drop=True)
delta_vws_sh = vws_sh.sel(time=slice('2001', '2020')).mean('time') - vws_sh.sel(time=slice('1980', '2000')).mean('time')
delta_u200_sh = u200_sh.sel(time=slice('2001', '2020')).mean('time') - u200_sh.sel(time=slice('1980', '2000')).mean('time')
delta_v200_sh = v200_sh.sel(time=slice('2001', '2020')).mean('time') - v200_sh.sel(time=slice('1980', '2000')).mean('time')
cf4 = ax4.contourf(delta_vws_sh.lon, delta_vws_sh.lat, delta_vws_sh, levels=np.linspace(-2, 2, 21), 
                   transform=ccrs.PlateCarree(), cmap=cmaps.MPL_RdBu_r, extend='both')
cbar4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', pad=0.15, aspect=50)
cbar4.ax.set_xlabel(r'$\mathrm{\Delta VWS\ (m\ s^{-1})}$', fontsize=18, labelpad=5)
cbar4.ax.set_xticks(np.arange(-2, 2.1, 0.4))
cbar4.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar4.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 3                   
lon2d, lat2d = np.meshgrid(delta_u200_sh.lon, delta_u200_sh.lat)
u_skip = delta_u200_sh.values[::skip, ::skip]
v_skip = delta_v200_sh.values[::skip, ::skip]
lon_skip = lon2d[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
q4 = ax4.quiver(lon_skip, lat_skip, u_skip, v_skip, scale=6, scale_units='inches',
                width=0.0015, headwidth=5, headlength=7, pivot='middle', transform=ccrs.PlateCarree())
qk4 = ax4.quiverkey(q4, 0.9, 1.05, 2, r'$\mathrm{2\ m\ s^{-1}}$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax4.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax4.set_aspect('equal')
ax4.set_xticks(np.arange(-180, 181, 60))
ax4.set_yticks(np.arange(-50, 51, 25))
ax4.set_xticks(np.arange(-180, 181, 20), minor=True)
ax4.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
ax4.set_yticklabels(['50°S', '25°S', '0°', '25°N', '50°N'])
ax4.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax4.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax4.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax4.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax4.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='grey', ls='--')
ax4.set_title(r'$\mathbf{e}$', fontsize=22, loc='left')
ax4.set_title('VWS & 200-hPa wind (DJFMAM)', fontsize=22)
for spine in ax4.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 


def calculate_significance(early_period, late_period):
    t_stat, p_val = stats.ttest_ind(late_period, early_period, axis=0, equal_var=False)
    return p_val
ax5 = fig.add_subplot(3, 2, 5)
colors = list(cmaps.NCV_blue_red(np.linspace(0, 1, 20)))
colors[9:11] = [(1,1,1,1), (1,1,1,1)]
cmap_white = ListedColormap(colors)
cf5 = ax5.contourf(W_nh.lat, W_nh.pressure, W_nh.mean('time'), levels=np.linspace(-40, 40, 21), cmap=cmap_white, extend='both')
cbar5 = fig.colorbar(cf5, ax=ax5, orientation='horizontal', pad=0.15, aspect=50)                
cbar5.ax.set_xlabel(r'$\mathrm{Vertical\ pressure\ velocity\ (hPa\ d^{-1})}$', fontsize=18, labelpad=5)
cbar5.ax.set_xticks(np.arange(-40, 42, 8))
cbar5.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar5.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 1              
lat2d, p2d = np.meshgrid(delta_V_nh.lat, delta_V_nh.pressure)
v_skip = delta_V_nh.values[::skip, ::skip]
w_skip = W_nh.values[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
p_skip = p2d[::skip, ::skip]
p_val_V_nh = calculate_significance(V_nh.sel(time=slice('1980', '2000')), V_nh.sel(time=slice('2001', '2020')))
p_val_W_nh = calculate_significance(W_nh.sel(time=slice('1980', '2000')), W_nh.sel(time=slice('2001', '2020')))
significance_mask = (p_val_V_nh < 0.05) | (p_val_W_nh < 0.05)
v_sig = np.where(significance_mask[::skip, ::skip], v_skip, np.nan)
w_sig = np.where(significance_mask[::skip, ::skip], w_skip, np.nan)
q5 = ax5.quiver(lat_skip, p_skip, v_skip, -100*w_skip, scale=0.8, scale_units='inches', color='k',
                width=0.002, headwidth=5, headlength=7, pivot='middle')
q5_sig = ax5.quiver(lat_skip, p_skip, v_sig, -100*w_sig, scale=0.8, scale_units='inches', color='g',
                    width=0.002, headwidth=5, headlength=7, pivot='middle')
qk5 = ax5.quiverkey(q5, 0.95, 1.05, 0.5, r'$0.3$', labelpos='E', coordinates='axes', fontproperties={'size': 15})
ax5.set_xlim([-30, 30])
ax5.set_xticklabels(['30°S', '20°S', '10°S', '0°', '10°N', '20°N', '30°N'])
ax5.set_ylim([1000, 105]) 
ax5.set_yscale('log')
ax5.set_yticks([1000, 700, 500, 300, 200, 105])
ax5.set_yticklabels([1000, 700, 500, 300, 200, 100])
ax5.minorticks_off()
ax5.set_ylabel('Pressure (hPa)', fontsize=20)
ax5.set_xticks(np.arange(-30, 31, 5), minor=True)
ax5.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax5.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax5.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax5.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax5.set_title(r'$\mathbf{c}$', fontsize=22, loc='left')
ax5.set_title('Hadley circulation (JJASON)', fontsize=22)
rect = patches.Rectangle((5, 106), 15, 885, ls='-', lw=2.5, ec='purple', fc='none', zorder=10)
ax5.add_patch(rect)
for spine in ax5.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 


ax6 = fig.add_subplot(3, 2, 6)
colors = list(cmaps.NCV_blue_red(np.linspace(0, 1, 20)))
colors[9:11] = [(1,1,1,1), (1,1,1,1)]
cmap_white = ListedColormap(colors)
cf6 = ax6.contourf(W_sh.lat, W_sh.pressure, W_sh.mean('time'), levels=np.linspace(-40, 40, 21), cmap=cmap_white, extend='both')
cbar6 = fig.colorbar(cf6, ax=ax6, orientation='horizontal', pad=0.15, aspect=50)                
cbar6.ax.set_xlabel(r'$\mathrm{Vertical\ pressure\ velocity\ (hPa\ d^{-1})}$', fontsize=18, labelpad=5)
cbar6.ax.set_xticks(np.arange(-40, 42, 8))
cbar6.ax.tick_params(axis='x', which='both', length=0, width=0)
for spine in cbar6.ax.spines.values():
    spine.set_linewidth(1.5)
skip = 1              
lat2d, p2d = np.meshgrid(delta_V_sh.lat, delta_V_sh.pressure)
v_skip = delta_V_sh.values[::skip, ::skip]
w_skip = delta_W_sh.values[::skip, ::skip]
lat_skip = lat2d[::skip, ::skip]
p_skip = p2d[::skip, ::skip]
p_val_V_sh = calculate_significance(V_sh.sel(time=slice('1980', '2000')), V_sh.sel(time=slice('2001', '2020')))
p_val_W_sh = calculate_significance(W_sh.sel(time=slice('1980', '2000')), W_sh.sel(time=slice('2001', '2020')))
significance_mask = (p_val_V_sh < 0.05) | (p_val_W_sh < 0.05)
v_sig = np.where(significance_mask[::skip, ::skip], v_skip, np.nan)
w_sig = np.where(significance_mask[::skip, ::skip], w_skip, np.nan)
q6 = ax6.quiver(lat_skip, p_skip, v_skip, -100*w_skip, scale=0.8, scale_units='inches', color='k',
                width=0.002, headwidth=5, headlength=7, pivot='middle')
q6_sig = ax6.quiver(lat_skip, p_skip, v_sig, -100*w_sig, scale=0.8, scale_units='inches', color='g',
                    width=0.002, headwidth=5, headlength=7, pivot='middle')
qk6 = ax6.quiverkey(q6, 0.95, 1.05, 0.5, r'$0.3$', labelpos='E', coordinates='axes', fontproperties={'size': 15})    
ax6.set_xlim([-30, 30])
ax6.set_xticklabels(['30°S', '20°S', '10°S', '0°', '10°N', '20°N', '30°N'])
ax6.set_ylim([1000, 105]) 
ax6.set_yscale('log')
ax6.set_yticks([1000, 700, 500, 300, 200, 105])
ax6.set_yticklabels([1000, 700, 500, 300, 200, 100])
ax6.minorticks_off()
ax6.set_ylabel('Pressure (hPa)', fontsize=20)
ax6.set_xticks(np.arange(-30, 31, 5), minor=True)
ax6.xaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax6.yaxis.set_tick_params(which='major', length=10, width=1.5, color='k', direction='out')
ax6.xaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax6.yaxis.set_tick_params(which='minor', length=5, width=1, color='k', direction='out')
ax6.set_title(r'$\mathbf{f}$', fontsize=22, loc='left')
ax6.set_title('Hadley circulation (DJFMAM)', fontsize=22)
rect = patches.Rectangle((-15, 106), 10, 885, ls='-', lw=2.5, ec='purple', fc='none', zorder=10)
ax6.add_patch(rect)
for spine in ax6.spines.values():
    spine.set_linewidth(1.5) 
    spine.set_color('k') 


for ax in (ax1, ax3):
    for MDR in MDRs:
        if MDR['lat_min'] >= 0:              
            rect = patches.Rectangle(
                (MDR['lon_min'], MDR['lat_min']),
                 MDR['lon_max'] - MDR['lon_min'],
                 MDR['lat_max'] - MDR['lat_min'],
                 ls='-', lw=2.5, ec='r', fc='none', transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)
for ax in (ax2, ax4):
    for MDR in MDRs:
        if MDR['lat_max'] <= 0:               
            rect = patches.Rectangle(
                (MDR['lon_min'], MDR['lat_min']),
                 MDR['lon_max'] - MDR['lon_min'],
                 MDR['lat_max'] - MDR['lat_min'],
                 ls='-', lw=2.5, ec='r', fc='none', transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)


plt.tight_layout()
# plt.savefig('Fig4.svg', bbox_inches='tight')
# plt.savefig('Fig4.pdf', bbox_inches='tight')