#!/usr/bin/env python3
"""Plot manuscript Fig.3 from ``svd_results.nc`` and ``svd_timeseries.csv``."""

from __future__ import annotations
import argparse
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

MDRS = (
    (120, 160, 5, 25), (240, 270, 5, 20), (310, 345, 5, 20),
    (55, 105, -15, -5), (150, 190, -20, -5), (60, 95, 5, 20),
)


def parse_args():
    p = argparse.ArgumentParser(description="Plot Fig.3")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "derived_data")
    p.add_argument("--output", type=Path, default=Path("Fig3.pdf"))
    p.add_argument("--dpi", type=int, default=500)
    return p.parse_args()


def setup_map(ax, title, panel, scf, add_boxes=False):
    ax.add_feature(cfeature.COASTLINE, edgecolor="k", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-50, 51, 25))
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticklabels(["50°S", "25°S", "0°", "25°N", "50°N"])
    ax.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color="grey", ls="--")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left")
    ax.set_title(title)
    ax.set_title(f"{scf:.0f}%", loc="right", fontsize=17)
    if add_boxes:
        for lon0, lon1, lat0, lat1 in MDRS:
            ax.add_patch(patches.Rectangle((lon0, lat0), lon1-lon0, lat1-lat0,
                                           ec="r", fc="none", lw=1.5,
                                           transform=ccrs.PlateCarree()))


def add_corr(ax, x, y, ypos, color):
    r, p = stats.pearsonr(x, y)
    ptxt = "P<0.001" if p < 0.001 else f"P={p:.3f}"
    ax.text(0.98, ypos, f"$r={r:.2f}$, {ptxt}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=15, color=color)


def main():
    a = parse_args()
    mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 18})
    ds = xr.open_dataset(a.data_dir / "svd_results.nc")
    ts = pd.read_csv(a.data_dir / "svd_timeseries.csv")
    scf = ds.scf.values * 100

    fig = plt.figure(figsize=(18, 12), dpi=a.dpi)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.18)

    for col, mode in enumerate((1, 2)):
        panel_map = "a" if mode == 1 else "d"
        panel_sst = "b" if mode == 1 else "e"
        panel_ts = "c" if mode == 1 else "f"

        ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree(central_longitude=180))
        z = ds.tcf_pattern.sel(mode=mode)
        ax.pcolormesh(ds.lon_tcf, ds.lat_tcf, z, cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                      transform=ccrs.PlateCarree())
        setup_map(ax, f"SVD{mode} TCF", panel_map, scf[mode-1], add_boxes=True)

        ax = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree(central_longitude=180))
        z = ds.sst_pattern.sel(mode=mode)
        pm = ax.pcolormesh(ds.lon_sst, ds.lat_sst, z, cmap="RdBu_r", vmin=-0.08, vmax=0.08,
                           transform=ccrs.PlateCarree())
        setup_map(ax, f"SVD{mode} SST", panel_sst, scf[mode-1])
        fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08, extend="both")

        ax = fig.add_subplot(gs[2, col])
        if mode == 1:
            ax.plot(ts.year, ts.EC1_TCF, lw=2, color="#4DBBD5", label=r"$EC1_{TCF}$")
            ax.plot(ts.year, ts.EC1_SST, lw=2, color="#E64B35", label=r"$EC1_{SST}$")
            ax.plot(ts.year, ts.global_mean_SST, lw=2, color="#3C5488", label="Global mean SST")
            add_corr(ax, ts.EC1_TCF, ts.global_mean_SST, 0.05, "#3C5488")
            title = "EC1 & Global mean SST"
        else:
            ax.plot(ts.year, ts.EC2_TCF, lw=2, color="#4DBBD5", label=r"$EC2_{TCF}$")
            ax.plot(ts.year, ts.EC2_SST, lw=2, color="#E64B35", label=r"$EC2_{SST}$")
            ax.plot(ts.year, ts.AMO, lw=2, color="#12AC1A", label="AMO index")
            ax.plot(ts.year, -ts.IPO, lw=2, color="purple", label="$-$IPO index")
            add_corr(ax, ts.EC2_TCF, ts.AMO, 0.18, "#12AC1A")
            add_corr(ax, ts.EC2_TCF, -ts.IPO, 0.05, "purple")
            title = "EC2 & Climate index"
        ax.legend(frameon=False, fontsize=15, ncol=2 if mode == 2 else 1)
        ax.axhline(0, color="grey", ls="--", lw=1.2)
        ax.set_xlim(ts.year.min(), ts.year.max())
        ax.set_title(rf"$\mathbf{{{panel_ts}}}$", loc="left")
        ax.set_title(title)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, bbox_inches="tight", dpi=a.dpi)
    plt.close(fig)
    ds.close()
    print(f"Figure written to: {a.output.resolve()}")


if __name__ == "__main__":
    main()