#!/usr/bin/env python3
"""Plot manuscript Fig.5 from ``environmental_fields.nc``.

Run ``01_prepare_data.py`` first.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap


MDRS = (
    ("WNP", 120, 160, 5, 25),
    ("ENP", 240, 270, 5, 20),
    ("NA", 310, 345, 5, 20),
    ("SI", 55, 105, -15, -5),
    ("SP", 150, 190, -20, -5),
    ("NI", 60, 95, 5, 20),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot manuscript Fig.5.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "derived_data",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Fig5.pdf",
    )
    p.add_argument("--dpi", type=int, default=500)
    return p.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
        }
    )


def style_map(ax) -> None:
    ax.add_feature(cfeature.COASTLINE, edgecolor="k", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-50, 51, 25))
    ax.set_xticks(np.arange(-180, 181, 20), minor=True)
    ax.set_xticklabels(
        ["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"]
    )
    ax.set_yticklabels(["50°S", "25°S", "0°", "25°N", "50°N"])

    ax.xaxis.set_tick_params(
        which="major", length=10, width=1.5, direction="out"
    )
    ax.yaxis.set_tick_params(
        which="major", length=10, width=1.5, direction="out"
    )
    ax.xaxis.set_tick_params(
        which="minor", length=5, width=1.0, direction="out"
    )
    ax.yaxis.set_tick_params(
        which="minor", length=5, width=1.0, direction="out"
    )

    ax.plot(
        [-180, 180],
        [0, 0],
        transform=ccrs.PlateCarree(),
        color="grey",
        linestyle="--",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("k")


def add_mdr_boxes(ax, hemisphere: str) -> None:
    for _, lon_min, lon_max, lat_min, lat_max in MDRS:
        if hemisphere == "NH" and lat_min < 0:
            continue
        if hemisphere == "SH" and lat_max > 0:
            continue

        ax.add_patch(
            patches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linestyle="-",
                linewidth=2.5,
                edgecolor="red",
                facecolor="none",
                transform=ccrs.PlateCarree(),
            )
        )


def add_horizontal_colorbar(
    fig,
    ax,
    artist,
    label: str,
    ticks,
) -> None:
    cbar = fig.colorbar(
        artist,
        ax=ax,
        orientation="horizontal",
        pad=0.15,
        aspect=50,
    )
    cbar.set_label(label, fontsize=18, labelpad=5)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(axis="x", length=0, width=0)
    cbar.outline.set_linewidth(1.5)


def plot_map_panel(
    fig,
    ax,
    shading: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    levels,
    cmap,
    colorbar_label: str,
    colorbar_ticks,
    title: str,
    panel: str,
    hemisphere: str,
    vector_scale: float,
    vector_key: float,
) -> None:
    style_map(ax)

    cf = ax.contourf(
        shading.lon,
        shading.lat,
        shading,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    skip = 3
    lon2d, lat2d = np.meshgrid(u.lon, u.lat)
    q = ax.quiver(
        lon2d[::skip, ::skip],
        lat2d[::skip, ::skip],
        u.values[::skip, ::skip],
        v.values[::skip, ::skip],
        scale=vector_scale,
        scale_units="inches",
        width=0.0015,
        headwidth=5,
        headlength=7,
        pivot="middle",
        transform=ccrs.PlateCarree(),
    )
    ax.quiverkey(
        q,
        0.90,
        1.05,
        vector_key,
        rf"${vector_key:g}\ \mathrm{{m\ s^{{-1}}}}$",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 15},
    )

    add_mdr_boxes(ax, hemisphere)
    add_horizontal_colorbar(
        fig, ax, cf, colorbar_label, colorbar_ticks
    )

    ax.set_title(rf"$\mathbf{{{panel}}}$", fontsize=22, loc="left")
    ax.set_title(title, fontsize=22)


def hadley_colormap():
    colors = list(cmaps.NCV_blue_red(np.linspace(0, 1, 20)))
    colors[9:11] = [(1, 1, 1, 1), (1, 1, 1, 1)]
    return ListedColormap(colors)


def plot_hadley_panel(
    fig,
    ax,
    ds: xr.Dataset,
    suffix: str,
    title: str,
    panel: str,
) -> None:
    w_clim = ds[f"hadley_w_clim_{suffix}"]
    delta_v = ds[f"hadley_delta_v_{suffix}"]
    delta_w = ds[f"hadley_delta_w_{suffix}"]
    significant = ds[f"hadley_significant_{suffix}"].astype(bool)

    cf = ax.contourf(
        w_clim.lat,
        w_clim.pressure,
        w_clim,
        levels=np.linspace(-40, 40, 21),
        cmap=hadley_colormap(),
        extend="both",
    )

    # Vector convention in the manuscript: meridional wind change and 100-fold pressure-velocity change. 
    # Positive vertical plotting direction corresponds to upward motion, hence the minus sign for omega.
    lat2d, p2d = np.meshgrid(delta_v.lat, delta_v.pressure)
    vv = delta_v.values
    ww = -100.0 * delta_w.values

    q = ax.quiver(
        lat2d,
        p2d,
        vv,
        ww,
        scale=0.8,
        scale_units="inches",
        color="k",
        width=0.002,
        headwidth=5,
        headlength=7,
        pivot="middle",
    )

    mask = significant.values
    ax.quiver(
        lat2d,
        p2d,
        np.where(mask, vv, np.nan),
        np.where(mask, ww, np.nan),
        scale=0.8,
        scale_units="inches",
        color="green",
        width=0.002,
        headwidth=5,
        headlength=7,
        pivot="middle",
    )

    ax.quiverkey(
        q,
        0.95,
        1.05,
        0.5,
        "0.5",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 15},
    )

    add_horizontal_colorbar(
        fig,
        ax,
        cf,
        r"Vertical pressure velocity (hPa d$^{-1}$)",
        np.arange(-40, 41, 8),
    )

    ax.set_xlim(-30, 30)
    ax.set_xticks(np.arange(-30, 31, 10))
    ax.set_xticklabels(
        ["30°S", "20°S", "10°S", "0°", "10°N", "20°N", "30°N"]
    )
    ax.set_xticks(np.arange(-30, 31, 5), minor=True)

    ax.set_ylim(1000, 105)
    ax.set_yscale("log")
    ax.set_yticks([1000, 700, 500, 300, 200, 105])
    ax.set_yticklabels([1000, 700, 500, 300, 200, 100])
    ax.minorticks_off()
    ax.set_ylabel("Pressure (hPa)", fontsize=20)

    ax.xaxis.set_tick_params(
        which="major", length=10, width=1.5, direction="out"
    )
    ax.yaxis.set_tick_params(
        which="major", length=10, width=1.5, direction="out"
    )

    # Purple boxes indicate the MDR latitude bands highlighted in the paper.
    if suffix == "nh":
        rect = patches.Rectangle(
            (5, 106), 15, 885,
            linestyle="-", linewidth=2.5,
            edgecolor="purple", facecolor="none", zorder=10,
        )
    else:
        rect = patches.Rectangle(
            (-15, 106), 10, 885,
            linestyle="-", linewidth=2.5,
            edgecolor="purple", facecolor="none", zorder=10,
        )
    ax.add_patch(rect)

    ax.set_title(rf"$\mathbf{{{panel}}}$", fontsize=22, loc="left")
    ax.set_title(title, fontsize=22)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("k")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    ds = xr.open_dataset(args.data_dir / "environmental_fields.nc")

    fig = plt.figure(figsize=(20, 13), dpi=args.dpi)

    # NH: a–c
    ax1 = fig.add_subplot(
        3, 2, 1, projection=ccrs.PlateCarree(central_longitude=180)
    )
    plot_map_panel(
        fig,
        ax1,
        ds.delta_sat_nh,
        ds.delta_u850_nh,
        ds.delta_v850_nh,
        levels=np.linspace(-1.5, 1.5, 21),
        cmap=cmaps.BlueWhiteOrangeRed,
        colorbar_label=r"$\Delta$ SAT (°C)",
        colorbar_ticks=np.arange(-1.5, 1.51, 0.3),
        title="SAT & 850-hPa wind (JJASON)",
        panel="a",
        hemisphere="NH",
        vector_scale=3,
        vector_key=1,
    )

    ax3 = fig.add_subplot(
        3, 2, 3, projection=ccrs.PlateCarree(central_longitude=180)
    )
    plot_map_panel(
        fig,
        ax3,
        ds.delta_vws_nh,
        ds.delta_u200_nh,
        ds.delta_v200_nh,
        levels=np.linspace(-2, 2, 21),
        cmap=cmaps.MPL_RdBu_r,
        colorbar_label=r"$\Delta$ VWS (m s$^{-1}$)",
        colorbar_ticks=np.arange(-2, 2.01, 0.4),
        title="VWS & 200-hPa wind (JJASON)",
        panel="b",
        hemisphere="NH",
        vector_scale=6,
        vector_key=2,
    )

    ax5 = fig.add_subplot(3, 2, 5)
    plot_hadley_panel(
        fig, ax5, ds, "nh", "Hadley circulation (JJASON)", "c"
    )

    # SH: d–f
    ax2 = fig.add_subplot(
        3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180)
    )
    plot_map_panel(
        fig,
        ax2,
        ds.delta_sat_sh,
        ds.delta_u850_sh,
        ds.delta_v850_sh,
        levels=np.linspace(-1.5, 1.5, 21),
        cmap=cmaps.BlueWhiteOrangeRed,
        colorbar_label=r"$\Delta$ SAT (°C)",
        colorbar_ticks=np.arange(-1.5, 1.51, 0.3),
        title="SAT & 850-hPa wind (DJFMAM)",
        panel="d",
        hemisphere="SH",
        vector_scale=3,
        vector_key=1,
    )

    ax4 = fig.add_subplot(
        3, 2, 4, projection=ccrs.PlateCarree(central_longitude=180)
    )
    plot_map_panel(
        fig,
        ax4,
        ds.delta_vws_sh,
        ds.delta_u200_sh,
        ds.delta_v200_sh,
        levels=np.linspace(-2, 2, 21),
        cmap=cmaps.MPL_RdBu_r,
        colorbar_label=r"$\Delta$ VWS (m s$^{-1}$)",
        colorbar_ticks=np.arange(-2, 2.01, 0.4),
        title="VWS & 200-hPa wind (DJFMAM)",
        panel="e",
        hemisphere="SH",
        vector_scale=6,
        vector_key=2,
    )

    ax6 = fig.add_subplot(3, 2, 6)
    plot_hadley_panel(
        fig, ax6, ds, "sh", "Hadley circulation (DJFMAM)", "f"
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    ds.close()

    print(f"Figure written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()