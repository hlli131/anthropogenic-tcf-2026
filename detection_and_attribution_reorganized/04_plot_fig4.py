#!/usr/bin/env python3
"""Plot manuscript Fig.4 from outputs of ``02_attribution_analysis.py``."""

from __future__ import annotations
import argparse
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

MDRS = (
    (120, 160, 5, 25), (240, 270, 5, 20), (310, 345, 5, 20),
    (55, 105, -15, -5), (150, 190, -20, -5), (60, 95, 5, 20),
)
COLORS = {"ALL": "#E64B35", "NAT": "#14AD96", "ANT": "#3C5488",
          "GHG": "#EC9E27", "AER": "#979595"}
REGIONS = ("Global", "NH", "SH")


def parse_args():
    p = argparse.ArgumentParser(description="Plot Fig.4")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "derived_data")
    p.add_argument("--output", type=Path, default=Path("Fig4.pdf"))
    p.add_argument("--agreement", type=int, default=12,
                   help="Minimum number of models agreeing on the trend sign")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def plot_map(fig, ax, ds, forcing, panel, threshold):
    trend, agree = ds[f"{forcing}_trend"], ds[f"{forcing}_agreement"]
    cmap = ListedColormap(cm.RdBu_r(np.linspace(0, 1, 12)))
    norm = BoundaryNorm(np.linspace(-0.09, 0.09, 13), 12)
    pm = ax.pcolormesh(trend.lon, trend.lat, trend, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree())
    lon2d, lat2d = np.meshgrid(trend.lon, trend.lat)
    mask = agree.values >= threshold
    ax.scatter(lon2d[mask], lat2d[mask], c="k", s=2, transform=ccrs.PlateCarree())
    for lon0, lon1, lat0, lat1 in MDRS:
        ax.add_patch(patches.Rectangle((lon0, lat0), lon1-lon0, lat1-lat0,
                                       ec="r", fc="none", lw=1.3,
                                       transform=ccrs.PlateCarree()))
    ax.add_feature(cfeature.COASTLINE, edgecolor="k", lw=0.7)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 181, 60)); ax.set_yticks(np.arange(-50, 51, 25))
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticklabels(["50°S", "25°S", "0°", "25°N", "50°N"])
    ax.plot([-180, 180], [0, 0], color="grey", ls="--", lw=1,
            transform=ccrs.PlateCarree())
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left"); ax.set_title(f"{forcing} simulation")
    cb = fig.colorbar(pm, ax=ax, orientation="vertical", pad=0.02, aspect=13)
    cb.set_label(r"decade$^{-1}$", rotation=270, labelpad=22)


def ordered_scaling(df):
    order = [
        ("1-signal", "ALL", 0), ("1-signal", "NAT", 1), ("1-signal", "ANT", 2),
        ("1-signal", "GHG", 3), ("1-signal", "AER", 4), ("2-signal", "NAT", 6),
        ("2-signal", "ANT", 7), ("3-signal", "NAT", 9), ("3-signal", "GHG", 10),
        ("3-signal", "AER", 11),
    ]
    rows = []
    for analysis, forcing, x in order:
        m = df[(df.analysis == analysis) & (df.forcing == forcing)]
        if not m.empty:
            row = m.iloc[0].to_dict(); row["x"] = x; rows.append(row)
    return pd.DataFrame(rows)


def plot_scaling(ax, all_df, region, panel):
    df = ordered_scaling(all_df[all_df.region == region])
    for _, r in df.iterrows():
        ax.bar(r.x, r.ci_upper-r.ci_lower, bottom=r.ci_lower, width=0.35,
               color=COLORS[r.forcing])
        ax.hlines(r.beta, r.x-0.175, r.x+0.175, color="white", lw=2)
    ax.axhline(0, color="grey", ls="--", lw=1); ax.axhline(1, color="grey", ls="--", lw=1)
    ax.axvline(5, color="k", lw=1); ax.axvline(8, color="k", lw=1)
    ax.set_xticks(df.x); ax.set_xticklabels(df.forcing, rotation=45)
    ax.set_ylabel("Scaling factor")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left"); ax.set_title(region)


def plot_attributable(ax, all_df, region, panel):
    df = all_df[all_df.region == region]
    obs = float(df.loc[df.forcing == "OBS", "trend"].iloc[0])
    forcings = ["ALL", "NAT", "ANT", "GHG", "AER"]
    stats_list = []
    for f in forcings:
        v = df.loc[df.forcing == f, "trend"].dropna().to_numpy()
        q5, q25, q75, q95 = np.percentile(v, [5, 25, 75, 95])
        stats_list.append({"label": f, "med": np.mean(v), "q1": q25, "q3": q75,
                           "whislo": q5, "whishi": q95, "fliers": []})
    b = ax.bxp(stats_list, showfliers=False, patch_artist=True, widths=0.5)
    for patch, f in zip(b["boxes"], forcings):
        patch.set_facecolor(COLORS[f]); patch.set_edgecolor("none")
    for med in b["medians"]:
        med.set_color("white"); med.set_linewidth(2)
    for cap in b["caps"]: cap.set_visible(False)
    ax.axhline(0, color="grey", ls="--", lw=1); ax.axhline(obs, color="k", lw=1.4)
    ax.set_ylabel(r"Attributable trend (decade$^{-1}$)")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left"); ax.set_title(region)


def main():
    a = parse_args()
    mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 16})
    maps = xr.open_dataset(a.data_dir / "mme_trends.nc")
    scaling = pd.read_csv(a.data_dir / "scaling_factors.csv")
    trends = pd.read_csv(a.data_dir / "attributable_trends.csv")

    fig = plt.figure(figsize=(20, 9), dpi=a.dpi)
    gs = fig.add_gridspec(3, 3, width_ratios=[5, 3.2, 1.8], wspace=0.35, hspace=0.35)
    for i, (f, p) in enumerate(zip(("ALL", "GHG", "AER"), "abc")):
        plot_map(fig, fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree(central_longitude=180)),
                 maps, f, p, a.agreement)
    for i, (r, p) in enumerate(zip(REGIONS, "def")):
        plot_scaling(fig.add_subplot(gs[i, 1]), scaling, r, p)
    for i, (r, p) in enumerate(zip(REGIONS, "ghi")):
        plot_attributable(fig.add_subplot(gs[i, 2]), trends, r, p)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, bbox_inches="tight", dpi=a.dpi)
    plt.close(fig); maps.close()
    print(f"Figure written to: {a.output.resolve()}")


if __name__ == "__main__":
    main()