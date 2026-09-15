#!/usr/bin/env python3
"""Prepare annual predictor data used for IML analysis.

Outputs
----------
IML_data.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import xarray as xr


MDRS = {
    "WNP": (120, 160, 5, 25),
    "ENP": (240, 270, 5, 20),
    "NA":  (310, 345, 5, 20),
    "SI":  (55, 105, -15, -5),
    "SP":  (150, 190, -20, -5),
    "NI":  (60, 95, 5, 20),
}

NH_MONTHS = (6, 7, 8, 9, 10, 11)
SH_MONTHS = (12, 1, 2, 3, 4, 5)

# Display name: (file name, NetCDF variable name)
VARIABLES = {
    "RH600": ("rh600.nc", "rh600"),
    "RV850": ("rv850.nc", "rv850"),
    "AV850": ("av850.nc", "av850"),
    "VWS":   ("VWS.nc", "vws"),
    "W500":  ("w500.nc", "w500"),
    "SST":   ("SST.nc", "sst"),
    "SSS":   ("SSS.nc", "sss"),
    "MLD":   ("MLDp03.nc", "mldp03"),
    "D26":   ("D26.nc", "d26"),
    "T100":  ("T100.nc", "t100"),
    "TCHP":  ("TCHP.nc", "tchp"),
    "MPI":   ("MPI.nc", "mpi"),
}

BASINS = tuple(MDRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IML input data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the IBTrACS-derived tcf time series and reanalysis data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_data" / "IML_data.csv",
        help="Output file for IML analysis.",
    )
    return parser.parse_args()


def _coord_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    """Return a slice that works for either ascending or descending coordinates."""
    return slice(lower, upper) if float(coord.values[0]) < float(coord.values[-1]) else slice(upper, lower)


def select_mdr(da: xr.DataArray, basin: str) -> xr.DataArray:
    """Select one MDR and its corresponding NH/SH TC-season months."""
    lon_min, lon_max, lat_min, lat_max = MDRS[basin]
    months = NH_MONTHS if lat_min > 0 else SH_MONTHS

    selected = da.sel(
        lon=_coord_slice(da["lon"], lon_min, lon_max),
        lat=_coord_slice(da["lat"], lat_min, lat_max),
    )
    return selected.where(selected["time.month"].isin(months), drop=True)


def annual_mean_mdr(da: xr.DataArray, basin: str) -> pd.Series:
    """Calculate annual means in TC MDRS."""
    selected = select_mdr(da, basin)
    mean_dims = [d for d in ("lat", "lon", "depth") if d in selected.dims]
    annual = selected.mean(dim=mean_dims).groupby("time.year").mean()
    return annual.to_series()


def load_predictor(path: Path, variable_name: str) -> xr.DataArray:
    """Load one predictor DataArray into memory and close the source file."""
    with xr.open_dataset(path) as ds:
        if variable_name not in ds:
            raise KeyError(f"{variable_name!r} not found in {path}")
        return ds[variable_name].load()


def build_IML_data(input_dir: Path) -> pd.DataFrame:
    tcf_path = input_dir / "tcf_timeseries.csv"
    if not tcf_path.exists():
        raise FileNotFoundError(tcf_path)

    table = pd.read_csv(tcf_path).rename(columns={"SEASON": "year"})
    if "year" not in table or "All" not in table:
        raise ValueError("tcf_timeseries.csv must contain 'SEASON' and 'All' columns.")

    for name, (filename, variable_name) in VARIABLES.items():
        da = load_predictor(input_dir / filename, variable_name)
        for basin in BASINS:
            series = annual_mean_mdr(da, basin).rename(f"{basin}_{name}")
            table = table.merge(series.rename_axis("year").reset_index(), on="year", how="left")

    # Equal-weight mean across the six MDRs. 
    # Cyclonic vorticity has opposite signs in the two hemispheres, so SH AV850/RV850 are sign-reversed first.
    for name in VARIABLES:
        cols = [f"{basin}_{name}" for basin in BASINS]
        values = table[cols].astype(float)
        if name in {"AV850", "RV850"}:
            values = values.mul([1, 1, 1, -1, -1, 1], axis=1)
        table[f"ALL_{name}"] = values.mean(axis=1)

    return table


def main() -> None:
    args = parse_args()
    table = build_IML_data(args.input_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(f"Years: {int(table.year.min())}–{int(table.year.max())}")
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()