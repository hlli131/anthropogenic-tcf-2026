#!/usr/bin/env python3
"""Prepare the environmental fields derived from DAMIP simulations.

The analysis compares two periods:
    P1 = 1980–2000
    P2 = 2001–2020

Northern Hemisphere TC season: JJASON (June–November)
Southern Hemisphere TC season: DJFMAM (December–May)

Inputs
----------
T2m.nc: t2m(time, lat, lon)
u850.nc: u850(time, lat, lon)
v850.nc: v850(time, lat, lon)
u200.nc: u200(time, lat, lon)
v200.nc: v200(time, lat, lon)
V.nc : v(time, pressure, lat, lon)
W.nc : w(time, pressure, lat, lon)

Output
----------
environmental_fields.nc
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import scipy.stats as stats
import xarray as xr


P1 = ("1980", "2000")
P2 = ("2001", "2020")
CLIM = ("1991", "2020")

NH_MONTHS = (6, 7, 8, 9, 10, 11)
SH_MONTHS = (12, 1, 2, 3, 4, 5)

FILES = {
    "sat": ("T2m_2p5.nc", "t2m"),
    "u850": ("u850_2p5.nc", "u850"),
    "v850": ("v850_2p5.nc", "v850"),
    "u200": ("u200_2p5.nc", "u200"),
    "v200": ("v200_2p5.nc", "v200"),
    "v_profile": ("V_2p5.nc", "v"),
    "w_profile": ("W_2p5.nc", "w"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DAMIP-based environmental fields.")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the processed monthly DAMIP simulations.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "derived_data",
    )
    return p.parse_args()


def load_field(input_dir: Path, filename: str, variable: str) -> xr.DataArray:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {path}")
        return ds[variable].load()


def seasonal(da: xr.DataArray, months: tuple[int, ...]) -> xr.DataArray:
    return da.where(da["time.month"].isin(months), drop=True)


def period_mean(
    da: xr.DataArray,
    start: str,
    end: str,
) -> xr.DataArray:
    return da.sel(time=slice(start, end)).mean("time", skipna=True)


def period_change(
    da: xr.DataArray,
    months: tuple[int, ...],
) -> xr.DataArray:
    """P2 minus P1 seasonal-mean change."""
    x = seasonal(da, months)
    return period_mean(x, *P2) - period_mean(x, *P1)


def calculate_vws(
    u850: xr.DataArray,
    v850: xr.DataArray,
    u200: xr.DataArray,
    v200: xr.DataArray,
) -> xr.DataArray:
    """Vertical wind shear magnitude between 850 and 200 hPa."""
    u850, v850, u200, v200 = xr.align(
        u850, v850, u200, v200, join="inner"
    )
    vws = np.hypot(u200 - u850, v200 - v850)
    vws.name = "vws"
    vws.attrs.update(
        {
            "long_name": "vertical wind shear magnitude between 850 and 200 hPa",
            "units": "m s-1",
        }
    )
    return vws


def normalize_pressure_coordinate(da: xr.DataArray) -> xr.DataArray:
    """Standardize the pressure coordinate to hPa when necessary."""
    if "pressure" not in da.coords:
        if "plev" in da.coords:
            da = da.rename({"plev": "pressure"})
        elif "level" in da.coords:
            da = da.rename({"level": "pressure"})
        else:
            raise ValueError(
                f"Expected a pressure/plev/level coordinate, got {list(da.coords)}"
            )

    pressure = da["pressure"].astype(float)
    if float(pressure.max()) > 2000:
        da = da.assign_coords(pressure=pressure / 100.0)
        da["pressure"].attrs["units"] = "hPa"

    return da.sortby("pressure", ascending=False)


def vertical_velocity_units(
    w: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return vertical pressure velocity in Pa s-1 and hPa day-1.

    CMIP/DAMIP ``wap`` is conventionally supplied in Pa s-1. 
    If the input explicitly identifies hPa day-1 instead, the inverse conversion is used.
    """
    units = str(w.attrs.get("units", "")).lower().replace(" ", "")

    is_hpa_day = "hpa" in units and ("day" in units or "d-1" in units)
    if is_hpa_day:
        w_hpa_day = w
        w_pa_s = w / 864.0
    else:
        if not units:
            warnings.warn(
                "W_2p5.nc has no units attribute; assuming Pa s-1.",
                stacklevel=2,
            )
        w_pa_s = w
        w_hpa_day = w * 864.0

    w_pa_s.attrs["units"] = "Pa s-1"
    w_hpa_day.attrs["units"] = "hPa day-1"
    return w_pa_s, w_hpa_day


def zonal_seasonal(
    da: xr.DataArray,
    months: tuple[int, ...],
) -> xr.DataArray:
    return seasonal(da, months).mean("lon", skipna=True)


def welch_p_value(
    early: xr.DataArray,
    late: xr.DataArray,
) -> xr.DataArray:
    """Conduct t-test over the time dimension."""
    early, late = xr.align(early, late, join="inner", exclude={"time"})

    result = stats.ttest_ind(
        late.values,
        early.values,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )

    dims = tuple(d for d in early.dims if d != "time")
    coords = {d: early[d] for d in dims}

    return xr.DataArray(
        result.pvalue,
        dims=dims,
        coords=coords,
        name="p_value",
    )


def hadley_diagnostics(
    v: xr.DataArray,
    w_pa_s: xr.DataArray,
    w_hpa_day: xr.DataArray,
    months: tuple[int, ...],
    suffix: str,
) -> dict[str, xr.DataArray]:
    """Prepare the zonal-mean Hadley-circulation diagnostics."""
    v_season = zonal_seasonal(v, months)
    w_pa_season = zonal_seasonal(w_pa_s, months)
    w_hd_season = zonal_seasonal(w_hpa_day, months)

    v_early = v_season.sel(time=slice(*P1))
    v_late = v_season.sel(time=slice(*P2))
    w_early = w_pa_season.sel(time=slice(*P1))
    w_late = w_pa_season.sel(time=slice(*P2))

    delta_v = v_late.mean("time") - v_early.mean("time")
    delta_w = w_late.mean("time") - w_early.mean("time")

    w_clim = w_hd_season.sel(time=slice(*CLIM)).mean("time")

    p_v = welch_p_value(v_early, v_late)
    p_w = welch_p_value(w_early, w_late)
    significant = ((p_v < 0.05) | (p_w < 0.05)).astype(np.int8)

    return {
        f"hadley_w_clim_{suffix}": w_clim,
        f"hadley_delta_v_{suffix}": delta_v,
        f"hadley_delta_w_{suffix}": delta_w,
        f"hadley_p_v_{suffix}": p_v,
        f"hadley_p_w_{suffix}": p_w,
        f"hadley_significant_{suffix}": significant,
    }


def main() -> None:
    args = parse_args()

    data = {
        key: load_field(args.input_dir, filename, variable)
        for key, (filename, variable) in FILES.items()
    }

    sat = data["sat"]
    u850 = data["u850"]
    v850 = data["v850"]
    u200 = data["u200"]
    v200 = data["v200"]

    vws = calculate_vws(u850, v850, u200, v200)

    v_profile = normalize_pressure_coordinate(data["v_profile"])
    w_profile = normalize_pressure_coordinate(data["w_profile"])
    v_profile, w_profile = xr.align(v_profile, w_profile, join="inner")

    w_pa_s, w_hpa_day = vertical_velocity_units(w_profile)

    fields: dict[str, xr.DataArray] = {}

    for suffix, months in (("nh", NH_MONTHS), ("sh", SH_MONTHS)):
        fields[f"delta_sat_{suffix}"] = period_change(sat, months)
        fields[f"delta_u850_{suffix}"] = period_change(u850, months)
        fields[f"delta_v850_{suffix}"] = period_change(v850, months)
        fields[f"delta_vws_{suffix}"] = period_change(vws, months)
        fields[f"delta_u200_{suffix}"] = period_change(u200, months)
        fields[f"delta_v200_{suffix}"] = period_change(v200, months)

        fields.update(
            hadley_diagnostics(
                v_profile,
                w_pa_s,
                w_hpa_day,
                months,
                suffix,
            )
        )

    ds = xr.Dataset(fields)
    ds.attrs.update(
        {
            "P1": "1980-2000",
            "P2": "2001-2020",
            "difference": "P2 minus P1",
            "Hadley_climatology": "1991-2020",
            "NH_TC_season": "JJASON",
            "SH_TC_season": "DJFMAM",
            "Hadley_significance": "t-test",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "environmental_fields.nc"
    ds.to_netcdf(output)

    print(f"Output written to: {output.resolve()}")


if __name__ == "__main__":
    main()