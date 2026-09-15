#!/usr/bin/env python3
"""Perform SVD analysis on the observed TCF and SST fields.

Inputs
----------
--tcf: annual gridded TCF data
--sst: annual SST data

Outputs
-----------
svd_results.nc: Paired TCF/SST patterns and squared covariance fractions.
svd_timeseries.csv: Standardized ECs, global-mean SST, AMO, and IPO indices.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description="SVD analysis")
    p.add_argument("--tcf", type=Path, required=True)
    p.add_argument("--sst", type=Path, required=True)
    p.add_argument("--sst-var", default="sst")
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "derived_data")
    return p.parse_args()


def open_da(path: Path, var: str | None = None) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if var is None:
            if len(ds.data_vars) != 1:
                raise ValueError(f"Specify a variable for {path}: {list(ds.data_vars)}")
            var = next(iter(ds.data_vars))
        return ds[var].load()


def annualize(da: xr.DataArray, start: int, end: int) -> xr.DataArray:
    dim = "time" if "time" in da.dims else "season"
    c = da[dim]
    if np.issubdtype(c.dtype, np.datetime64):
        da = da.sel({dim: slice(str(start), str(end))})
        years = da[dim].dt.year
        if len(np.unique(years)) < da.sizes[dim]:
            da = da.groupby(f"{dim}.year").mean(dim)
        else:
            da = da.assign_coords({dim: years.values}).rename({dim: "year"})
    else:
        da = da.sel({dim: slice(start, end)}).rename({dim: "year"})
    da = da.assign_coords(year=da.year.astype(int))
    return da.sel(year=slice(start, end))


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def flatten_complete(da):
    a = da.transpose("year", "lat", "lon").values
    f = a.reshape(a.shape[0], -1)
    valid = np.isfinite(f).all(axis=0)
    return f[:, valid], valid


def paired_svd(tcf, sst):
    x, xv = flatten_complete(tcf)
    y, yv = flatten_complete(sst)
    x -= x.mean(0, keepdims=True)
    y -= y.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x.T @ y / (len(x) - 1), full_matrices=False)
    v = vt.T
    scf = s**2 / np.sum(s**2)

    def restore(vec, valid, template):
        a = np.full(template.sizes["lat"] * template.sizes["lon"], np.nan)
        a[valid] = vec
        return a.reshape(template.sizes["lat"], template.sizes["lon"])

    tp = np.stack([restore(u[:, i], xv, tcf) for i in range(2)])
    sp = np.stack([restore(v[:, i], yv, sst) for i in range(2)])
    return tp, sp, x @ u[:, :2], y @ v[:, :2], scf[:2]


def weighted_mean(da, lat0=-60, lat1=60):
    sl = slice(lat1, lat0) if da.lat[0] > da.lat[-1] else slice(lat0, lat1)
    sub = da.sel(lat=sl)
    return sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(("lat", "lon"))


def amo_index(sst):
    lat_sl = slice(60, 0) if sst.lat[0] > sst.lat[-1] else slice(0, 60)
    if float(sst.lon.min()) >= 0:
        atl = sst.sel(lat=lat_sl, lon=slice(280, 360))
    else:
        atl = sst.sel(lat=lat_sl, lon=slice(-80, 0))
    atl_mean = atl.weighted(np.cos(np.deg2rad(atl.lat))).mean(("lat", "lon"))
    return atl_mean - weighted_mean(sst)


def ipo_index(sst):
    """PC2 of 13-year low-pass-filtered SST."""
    smooth = sst.rolling(year=13, center=True, min_periods=7).mean()
    a, valid = flatten_complete(smooth)
    a -= a.mean(0, keepdims=True)
    lat2d = np.repeat(sst.lat.values[:, None], sst.sizes["lon"], axis=1).ravel()
    a *= np.sqrt(np.cos(np.deg2rad(lat2d[valid])))[None, :]
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, 1] * s[1]


def orient(tp, sp, et, es, gm, amo):
    for mode, ref in ((0, gm), (1, amo)):
        if stats.pearsonr(et[:, mode], ref)[0] < 0:
            tp[mode] *= -1
            sp[mode] *= -1
            et[:, mode] *= -1
            es[:, mode] *= -1


def main():
    a = parse_args()
    tcf = annualize(open_da(a.tcf), a.start_year, a.end_year)
    sst = annualize(open_da(a.sst, a.sst_var), a.start_year, a.end_year)
    years = np.intersect1d(tcf.year, sst.year)
    tcf, sst = tcf.sel(year=years), sst.sel(year=years)

    gm = zscore(weighted_mean(sst).values)
    amo = zscore(amo_index(sst).values)
    ipo = zscore(ipo_index(sst))
    tp, sp, et, es, scf = paired_svd(tcf, sst)
    orient(tp, sp, et, es, gm, amo)
    et = np.column_stack([zscore(et[:, i]) for i in range(2)])
    es = np.column_stack([zscore(es[:, i]) for i in range(2)])

    out = xr.Dataset(
        {
            "tcf_pattern": (("mode", "lat_tcf", "lon_tcf"), tp),
            "sst_pattern": (("mode", "lat_sst", "lon_sst"), sp),
            "scf": ("mode", scf),
        },
        coords={
            "mode": [1, 2], "lat_tcf": tcf.lat, "lon_tcf": tcf.lon,
            "lat_sst": sst.lat, "lon_sst": sst.lon,
        },
    )
    ts = pd.DataFrame({
        "year": years.astype(int), "EC1_TCF": et[:, 0], "EC1_SST": es[:, 0],
        "EC2_TCF": et[:, 1], "EC2_SST": es[:, 1], "global_mean_SST": gm,
        "AMO": amo, "IPO": ipo,
    })

    a.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.output_dir / "svd_results.nc")
    ts.to_csv(a.output_dir / "svd_timeseries.csv", index=False)
    print(f"Outputs written to: {a.output_dir.resolve()}")


if __name__ == "__main__":
    main()