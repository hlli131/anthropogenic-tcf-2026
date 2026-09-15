#!/usr/bin/env python3
"""Prepare observed TCF data and trend statistics.

Outputs
----------
tcf_anomalies.csv
timeseries_trends.csv
spatial_trends_2p5.nc
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


START_YEAR, END_YEAR = 1980, 2020
GRID_RESOLUTION = 2.5
BASINS = ("WNP", "ENP", "NA", "SI", "SP", "NI")
REGIONS = ("Global", "NH", "SH", *BASINS)
EXTRA_COLUMNS = {"USA": "SSHS", "CMA": "CAT", "TOKYO": "GRADE"}
NH_MONTHS = (6, 7, 8, 9, 10, 11)
SH_MONTHS = (12, 1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare observed TCF data.")
    parser.add_argument("--ibtracs", type=Path, required=True, help="Path to IBTrACS CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "derived_data",
        help="Output directory (default: ./derived_data).",
    )
    parser.add_argument("--agency", default="USA", help="IBTrACS agency prefix.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    
    return parser.parse_args()


def load_ibtracs(
    path: Path,
    agency: str = "USA",
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """Read and filter IBTrACS."""
    agency = agency.upper()
    lat_col, lon_col = f"{agency}_LAT", f"{agency}_LON"
    columns = [
        "SID", "SEASON", "BASIN", "NAME", "ISO_TIME",
        lat_col, lon_col, f"{agency}_WIND", f"{agency}_PRES",
    ]
    
    if agency in EXTRA_COLUMNS:
        columns.append(f"{agency}_{EXTRA_COLUMNS[agency]}")

    data = pd.read_csv(path, keep_default_na=False, low_memory=False)
    data = data[data["NATURE"].eq("TS")][columns].copy()
    data["SEASON"] = data["SEASON"].astype(int)
    data = data[data["SEASON"].between(start_year, end_year)]
    data = data[~data["BASIN"].eq("SA")]
    data["BASIN"] = data["BASIN"].replace({"WP": "WNP", "EP": "ENP"})
    data = data[data["NAME"] != "UNNAMED"]
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"])
    data = data[data["ISO_TIME"].dt.hour.isin((0, 6, 12, 18))]
    data = data[data[f"{agency}_WIND"] != " "]
    data[["SID", "BASIN", "NAME"]] = data[["SID", "BASIN", "NAME"]].astype("string")
    data[[lat_col, lon_col]] = data[[lat_col, lon_col]].astype(float)
    
    return data.reset_index(drop=True)


def select_first_ts_record(data: pd.DataFrame, agency: str = "USA") -> pd.DataFrame:
    """Keep the first TS record of each SID."""
    agency = agency.upper()
    lat_col, lon_col = f"{agency}_LAT", f"{agency}_LON"
    genesis = data.drop_duplicates("SID", keep="first").copy()
    genesis[lon_col] = np.where(genesis[lon_col] >= 0, genesis[lon_col], genesis[lon_col] + 360)
    genesis["Hemisphere"] = np.where(genesis[lat_col] > 0, "NH", "SH")
    genesis["MONTH"] = genesis["ISO_TIME"].dt.month.astype(int)

    return genesis.reset_index(drop=True)


def calculate_annual_counts(
    genesis: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Calculate annual Global, hemispheric, and basin TCF counts."""
    years = pd.Index(range(start_year, end_year + 1), name="SEASON")
    result = pd.DataFrame(index=years)
    result["Global"] = genesis.groupby("SEASON")["SID"].nunique()
    hemi = genesis.groupby(["SEASON", "Hemisphere"])["SID"].nunique().unstack(fill_value=0)
    basin = genesis.groupby(["SEASON", "BASIN"])["SID"].nunique().unstack(fill_value=0)
    result = result.join(hemi.reindex(columns=("NH", "SH"), fill_value=0))
    result = result.join(basin.reindex(columns=BASINS, fill_value=0))
    result = result.fillna(0).astype(int).reset_index()

    return result[["SEASON", *REGIONS]]


def calculate_anomalies(counts: pd.DataFrame) -> pd.DataFrame:
    anomalies = pd.DataFrame({"SEASON": counts["SEASON"]})
    values = counts[list(REGIONS)].astype(float)
    anomalies[list(REGIONS)] = values - values.mean(axis=0)

    return anomalies


def calculate_timeseries(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual TCF trends for all regions."""
    x = anomalies["SEASON"].to_numpy(float)
    rows = []
    
    for region in REGIONS:
        fit = stats.linregress(x, anomalies[region].to_numpy(float))
        rows.append((region, fit.slope, fit.intercept, fit.pvalue, fit.stderr))

    return pd.DataFrame(
        rows,
        columns=("region", "slope_per_year", "intercept", "p_value", "stderr_per_year"),
    )


def calculate_monthly_gridded_tcf(
    genesis: pd.DataFrame,
    agency: str,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """Count genesis records on the 2.5-degree grid."""
    agency = agency.upper()
    lat_col, lon_col = f"{agency}_LAT", f"{agency}_LON"
    lat_bins = np.arange(91.25, -91.25, -GRID_RESOLUTION)
    lon_bins = np.arange(-1.25, 361.0, GRID_RESOLUTION)
    months = np.sort(genesis["MONTH"].unique())
    n_months = len(months)
    shape = (
        (end_year - start_year + 1) * n_months,
        len(lat_bins) - 1,
        len(lon_bins) - 1,
    )
    fields = np.zeros(shape, dtype=np.float32)
    lat_idx = np.digitize(genesis[lat_col].to_numpy(float), lat_bins) - 1
    lon_idx = np.digitize(genesis[lon_col].to_numpy(float), lon_bins) - 1
    month_idx = np.searchsorted(months, genesis["MONTH"].to_numpy())
    time_idx = (genesis["SEASON"].to_numpy() - start_year) * n_months + month_idx
    valid = (
        (time_idx >= 0) & (time_idx < shape[0])
        & (lat_idx >= 0) & (lat_idx < shape[1])
        & (lon_idx >= 0) & (lon_idx < shape[2])
    )
    np.add.at(fields, (time_idx[valid], lat_idx[valid], lon_idx[valid]), 1)
    
    return xr.DataArray(
        fields,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range(f"{start_year}-01-01", periods=shape[0], freq="MS"),
            "lat": np.arange(88.75, -88.75 - GRID_RESOLUTION, -GRID_RESOLUTION),
            "lon": np.arange(1.25, 358.75 + GRID_RESOLUTION, GRID_RESOLUTION),
        },
        name="tcf",
        attrs={"name": "monthly tropical cyclone frequency", "units": "unitless"},
    )


def aggregate_annual_gridded_tcf(monthly_tcf: xr.DataArray) -> xr.DataArray:
    """Aggregate NH JJASON and SH DJFMAM."""
    month = monthly_tcf.time.dt.month
    mask = xr.where(
        monthly_tcf.lat > 0,
        month.isin(NH_MONTHS),
        month.isin(SH_MONTHS),
    )
    annual = monthly_tcf.where(mask).groupby("time.year").sum("time", keep_attrs=True)
    annual = annual.rename({"year": "time"})
    annual = annual.assign_coords(time=pd.to_datetime(annual.time.values.astype(str)))
    annual.name = "annual_tcf"
    annual.attrs.update(name="annual tropical cyclone frequency", units="count")
    
    return annual


def calculate_spatial_trends(annual_tcf: xr.DataArray) -> xr.Dataset:
    """Calculate grid-cell trends and p-values."""
    x = np.arange(annual_tcf.sizes["time"], dtype=float)
    
    def regress(y: np.ndarray) -> tuple[float, float]:
        fit = stats.linregress(x, y)
        
        return fit.slope * 10.0, fit.pvalue

    trend, p_value = xr.apply_ufunc(
        regress,
        annual_tcf,
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float],
    )
    trend = trend.where(trend != 0)
    
    return xr.Dataset(
        {
            "trend_decade": trend.assign_attrs(units="count decade-1"),
            "p_value": p_value.assign_attrs(units="1"),
        }
    )


def save_outputs(
    output_dir: Path,
    anomalies: pd.DataFrame,
    timeseries: pd.DataFrame,
    spatial_trends: xr.Dataset,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(output_dir / "tcf_anomalies.csv", index=False)
    timeseries.to_csv(output_dir / "timeseries_trends.csv", index=False)
    spatial_trends.to_netcdf(output_dir / "spatial_trends_2p5.nc")


def main() -> None:
    args = parse_args()
    records = load_ibtracs(args.ibtracs, args.agency, args.start_year, args.end_year)
    genesis = select_first_ts_record(records, args.agency)
    counts = calculate_annual_counts(genesis, args.start_year, args.end_year)
    anomalies = calculate_anomalies(counts)
    timeseries = calculate_timeseries(anomalies)
    monthly_tcf = calculate_monthly_gridded_tcf(genesis, args.agency, args.start_year, args.end_year)
    annual_tcf = aggregate_annual_gridded_tcf(monthly_tcf)
    spatial_trends = calculate_spatial_trends(annual_tcf)
    save_outputs(args.output_dir, anomalies, timeseries, spatial_trends)
    
    print(f"Selected genesis records: {len(genesis)}")
    print(f"Analysis period: {args.start_year}-{args.end_year}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()