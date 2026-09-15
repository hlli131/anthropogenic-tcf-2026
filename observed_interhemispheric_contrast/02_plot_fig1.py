#!/usr/bin/env python3
"""Plot manuscript Fig.1 from the derived observational TCF data.

Run ``01_analyze_observed_tcf.py`` first.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MultipleLocator


MDRS = (
    ("WNP", 120, 160, 5, 25),
    ("ENP", 240, 270, 5, 20),
    ("NA", 310, 345, 5, 20),
    ("SI", 55, 105, -15, -5),
    ("SP", 150, 190, -20, -5),
    ("NI", 60, 95, 5, 20),
)

TIME_SERIES_PANELS = (
    ("Global", "darkblue", (-20, 20), "b"),
    ("NH", "green", (-20, 20), "c"),
    ("SH", "purple", (-10, 10), "d"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig.1 from derived TCF data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "derived_data",
        help="Directory produced by 01_analyze_observed_tcf.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Fig1.pdf",
        help="Output figure path (default: ./Fig1.pdf).",
    )
    parser.add_argument("--dpi", type=int, default=500)
    
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
        }
    )


def format_p_value(p: float) -> str:
    if p < 0.001:
        return r"$p<0.001$"
    return rf"$P={p:.3f}$"


def style_timeseries_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    ax.axhline(y=0, color="k", alpha=0.5, lw=1.5, linestyle="--")
    ax.set_ylim(*ylim)
    ax.set_xlim(1980, 2020)
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_tick_params(which="major", length=6, width=1.5, direction="out", pad=6)
    ax.xaxis.set_tick_params(which="minor", length=4, width=1.5, direction="out", pad=6)
    ax.yaxis.set_tick_params(which="major", length=6, width=1.5, direction="out", pad=4)
    ax.yaxis.set_tick_params(which="minor", length=4, width=1.5, direction="out", pad=4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_map(
    fig: plt.Figure,
    ax: plt.Axes,
    spatial_trends: xr.Dataset,
) -> None:
    ax.add_feature(cfeature.COASTLINE, edgecolor="k", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="white")
    cmap = ListedColormap(cm.RdBu_r(np.linspace(0, 1, 12)))
    boundaries = np.linspace(-0.15, 0.15, 13)
    norm = BoundaryNorm(boundaries=boundaries, ncolors=12, extend="neither")
    trend = spatial_trends["trend_decade"]
    p_value = spatial_trends["p_value"]
    pmesh = ax.pcolormesh(
        trend.lon,
        trend.lat,
        trend,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )
    lons, lats = np.meshgrid(trend.lon.to_numpy(), trend.lat.to_numpy())
    significant = p_value.to_numpy() < 0.15
    ax.scatter(
        lons[significant],
        lats[significant],
        marker="o",
        s=6,
        c="k",
        transform=ccrs.PlateCarree(),
    )
    
    for _, lon_min, lon_max, lat_min, lat_max in MDRS:
        ax.add_patch(
            patches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linestyle="-",
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                transform=ccrs.PlateCarree(),
            )
        )
    
    cbar = fig.colorbar(
        pmesh,
        ax=ax,
        orientation="horizontal",
        pad=0.07,
        fraction=0.1,
        aspect=50,
    )
    cbar.ax.set_xlabel(r"$\mathrm{Annual\ TCF\ trend\ (decade^{-1})}$", fontsize=20, labelpad=10)
    cbar.ax.grid(True, which="both", axis="both", lw=1.5, linestyle="-", color="k")
    cbar.ax.tick_params(axis="x", which="both", length=0, width=0)
    
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.set_aspect("equal")
    ax.set_title(r"$\mathbf{a}$", fontsize=22, loc="left")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-50, 51, 25))
    ax.set_xticks(np.arange(-180, 181, 20), minor=True)
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticklabels(["50°S", "25°S", "0°", "25°N", "50°N"])
    ax.xaxis.set_tick_params(which="major", length=10, width=1.5, direction="out", labelsize=18)
    ax.yaxis.set_tick_params(which="major", length=10, width=1.5, direction="out", labelsize=18)
    ax.xaxis.set_tick_params(which="minor", length=6, width=1.5, direction="out")
    ax.yaxis.set_tick_params(which="minor", length=6, width=1.5, direction="out")
    ax.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color="grey", linestyle="--")
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("k")


def plot_time_series(
    ax: plt.Axes,
    anomalies: pd.DataFrame,
    trend_stats: pd.DataFrame,
    region: str,
    color: str,
    ylim: tuple[float, float],
    panel: str,
) -> None:
    years = anomalies["SEASON"].to_numpy(dtype=float)
    values = anomalies[region].to_numpy(dtype=float)
    stats_row = trend_stats.loc[trend_stats["region"].eq(region)].iloc[0]
    slope = float(stats_row["slope_per_year"])
    intercept = float(stats_row["intercept"])
    p_value = float(stats_row["p_value"])
    ax.plot(years, values, color=color, lw=1.5, alpha=0.5)
    ax.plot(years, intercept + slope * years, linestyle="--", color=color, lw=2)
    smoothed = pd.Series(values).rolling(window=5, center=True).mean().to_numpy()
    ax.plot(years, smoothed, linestyle="-", color=color, lw=1.5)
    ax.text(
        0.97,
        0.04,
        rf"$\mathrm{{Trend={slope:.2f}}}$" + "\n" + format_p_value(p_value),
        fontsize=16,
        color="k",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        linespacing=1.5
    )
    style_timeseries_axis(ax, ylim)
    ax.set_title(rf"$\mathbf{{{panel}}}$", fontsize=22, loc="left")
    ax.set_title(region, fontsize=22)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    spatial_trends = xr.open_dataset(args.data_dir / "spatial_trends_2p5.nc")
    anomalies = pd.read_csv(args.data_dir / "tcf_anomalies.csv")
    trend_stats = pd.read_csv(args.data_dir / "timeseries_trends.csv")
    fig = plt.figure(figsize=(15, 12), dpi=args.dpi)
    gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
    ax_map = plt.subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(fig, ax_map, spatial_trends)
    axes = [plt.subplot(gs[1, i]) for i in range(3)]
    
    for ax, (region, color, ylim, panel) in zip(axes, TIME_SERIES_PANELS, strict=True):
        plot_time_series(ax, anomalies, trend_stats, region, color, ylim, panel)
    
    axes[0].set_ylabel("TCF anomaly", fontsize=18)
    fig.tight_layout(w_pad=1, h_pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    spatial_trends.close()
    
    print(f"Figure written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()