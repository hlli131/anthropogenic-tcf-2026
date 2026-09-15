#!/usr/bin/env python3
"""DAMIP detection-and-attribution analysis using PyDnA.
* ``PyDnA.regC``              : Ledoit-Wolf-style regularized covariance
* ``PyDnA.projfullrank``      : temporal centering/full-rank projection
* ``PyDnA.tls``               : TLS scaling factors and 90% confidence intervals
* ``PyDnA.consist_mc_tls``    : Monte-Carlo residual-consistency null distribution
* ``PyDnA.gke``               : kernel-estimated residual-consistency p value

Inputs
----------
observed_tcf.nc: tcf(time, lat, lon)
forced_tcf.nc: tcf(forcing, model, member, time, lat, lon)
picontrol_tcf.nc: tcf(model, member, time, lat, lon)

Outputs
----------
mme_trends.nc: MME trend maps and intermodel sign-agreement counts for ALL/GHG/AER.
scaling_factors.csv: One-, two-, and three-signal TLS scaling factors and 90% CIs.
residual_consistency.csv: Residual-consistency statistic and p-value for each regression.
attributable_trends.csv: Attributable trends from the one-signal scaling factors.
fingerprint_ensemble_sizes.csv: Effective ensemble sizes supplied to the PyDnA TLS noise normalization.
"""


from __future__ import annotations
import argparse
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import PyDnA as pda


FORCINGS = ("ALL", "NAT", "GHG", "AER")
REGIONS = ("Global", "NH", "SH")
ATTRIBUTABLE_FORCINGS = ("ALL", "NAT", "ANT", "GHG", "AER")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regularized TLS optimal fingerprinting using PyDnA."
    )
    p.add_argument("--observed", type=Path, required=True)
    p.add_argument("--forced", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--var", default="tcf")
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument(
        "--block-years",
        type=int,
        default=5,
        help="Length of non-overlapping temporal means used in optimal fingerprinting.",
    )
    p.add_argument(
        "--tls-ci",
        choices=("AS03", "ODP"),
        default="AS03",
        help="TLS confidence-interval formula (default: AS03).",
    )
    p.add_argument(
        "--consistency",
        choices=("MC", "AS03"),
        default="MC",
        help="Residual-consistency p-value method (default: Monte Carlo).",
    )
    p.add_argument(
        "--n-consistency",
        type=int,
        default=5000,
        help="Monte-Carlo runs for the residual-consistency test.",
    )
    p.add_argument(
        "--control-split",
        choices=("alternating", "random", "segment"),
        default="alternating",
        help="How independent piControl realizations are divided into Z1 and Z2.",
    )
    p.add_argument("--seed", type=int, default=1577)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "derived_data",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# PyDnA compatibility helpers
# -----------------------------------------------------------------------------
@contextmanager
def pydna_tls_numpy_compat(pda):
    """Small compatibility shim for the 2020 PyDnA TLS routine.

    PyDnA's multi-signal CI code passes lists containing 1x1 ``np.matrix`` objects to ``np.max``/``np.argmax``. 
    Recent NumPy versions no longer coerce those lists automatically.
    We scalarize only that narrow case while ``PyDnA.tls`` is running; no PyDnA scientific formula is changed.
    """
    old_max = pda.np.max
    old_argmax = pda.np.argmax

    def scalarize_list(values):
        if not isinstance(values, (list, tuple)):
            return values
        out = []
        for value in values:
            arr = np.asarray(value)
            if arr.size != 1:
                return values
            out.append(float(arr.ravel()[0]))
        return np.asarray(out)

    def safe_max(values, *args, **kwargs):
        return old_max(scalarize_list(values), *args, **kwargs)

    def safe_argmax(values, *args, **kwargs):
        return old_argmax(scalarize_list(values), *args, **kwargs)

    pda.np.max = safe_max
    pda.np.argmax = safe_argmax
    try:
        yield
    finally:
        pda.np.max = old_max
        pda.np.argmax = old_argmax


@contextmanager
def legacy_numpy_seed(seed: int):
    """Temporarily seed NumPy's global RNG because PyDnA uses it internally."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def real_matrix(a: np.ndarray, name: str) -> np.ndarray:
    """Discard negligible numerical imaginary parts from sqrtm/inverse results."""
    a = np.asarray(a)
    imag = np.max(np.abs(np.imag(a)))
    scale = max(1.0, float(np.max(np.abs(np.real(a)))))
    if imag > 1e-8 * scale:
        raise ValueError(f"{name} has a non-negligible imaginary component.")
    return np.real(a)


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------
def load_var(path: Path, var: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError(f"{var!r} not found in {path}")
        return ds[var].load()


def time_years(da: xr.DataArray) -> np.ndarray:
    if np.issubdtype(da.time.dtype, np.datetime64):
        return da.time.dt.year.values.astype(int)
    return np.asarray(da.time.values, dtype=int)


def select_period(da: xr.DataArray, start: int, end: int) -> xr.DataArray:
    y = time_years(da)
    return da.isel(time=np.where((y >= start) & (y <= end))[0])


def region_total(da: xr.DataArray, region: str) -> xr.DataArray:
    """Regional TCF total used by the Global/NH/SH fingerprinting analyses."""
    if region == "NH":
        da = da.where(da.lat > 0, drop=True)
    elif region == "SH":
        da = da.where(da.lat < 0, drop=True)
    elif region != "Global":
        raise ValueError(f"Unknown region: {region}")

    return da.sum(("lat", "lon"), skipna=True)


def model_mean(da: xr.DataArray) -> xr.DataArray:
    """Average ensemble members within each model and preserve equal model weights."""
    return da.mean("member", skipna=True) if "member" in da.dims else da


def ensure_model_dim(da: xr.DataArray) -> xr.DataArray:
    return da if "model" in da.dims else da.expand_dims(model=["single_model"])


def forcing_model_series(
    forced: xr.DataArray,
    forcing: str,
    region: str,
) -> xr.DataArray:
    """Per-model ensemble-mean regional TCF time series."""
    return ensure_model_dim(
        model_mean(region_total(forced.sel(forcing=forcing), region))
    )


def ant_model_series(forced: xr.DataArray, region: str) -> xr.DataArray:
    """ANT = ALL - NAT at the model-ensemble-mean level."""
    all_ = forcing_model_series(forced, "ALL", region)
    nat = forcing_model_series(forced, "NAT", region)
    all_, nat = xr.align(all_, nat, join="inner")
    return all_ - nat


def mme_series(forced: xr.DataArray, region: str) -> dict[str, xr.DataArray]:
    """Equal-model-weight multimodel ensemble fingerprints."""
    out = {
        forcing: forcing_model_series(forced, forcing, region).mean("model", skipna=True)
        for forcing in FORCINGS
    }
    out["ANT"] = ant_model_series(forced, region).mean("model", skipna=True)
    return out


def annual_series(da: xr.DataArray) -> pd.Series:
    """Convert a one-dimensional annual DataArray to a year-indexed Series."""
    if da.ndim != 1 or da.dims[0] != "time":
        raise ValueError(f"Expected 1-D time series, got dims={da.dims}")
    return pd.Series(np.asarray(da.values, float), index=time_years(da))


def make_block_fingerprints(
    obs: xr.DataArray,
    signals: dict[str, xr.DataArray],
    start_year: int,
    end_year: int,
    block_years: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Align annual series and form non-overlapping block means. """
    series = {"OBS": annual_series(obs)}
    series.update({name: annual_series(da) for name, da in signals.items()})

    years = set(range(start_year, end_year + 1))
    for s in series.values():
        years &= set(int(y) for y in s.dropna().index)
    years = np.asarray(sorted(years), dtype=int)

    n_blocks = len(years) // block_years
    if n_blocks < 3:
        raise ValueError("Too few complete temporal blocks for fingerprinting.")

    n_used = n_blocks * block_years
    years = years[:n_used]

    # Require consecutive annual data so a block has an unambiguous meaning.
    if not np.all(np.diff(years) == 1):
        raise ValueError("Common observation/model years are not consecutive.")

    labels = [
        f"{years[i]:d}-{years[i + block_years - 1]:d}"
        for i in range(0, n_used, block_years)
    ]

    def block(s: pd.Series) -> np.ndarray:
        v = s.loc[years].to_numpy(float)
        return v.reshape(n_blocks, block_years).mean(axis=1)

    y = block(series["OBS"])
    x = {name: block(series[name]) for name in signals}
    return y, x, labels


# -----------------------------------------------------------------------------
# Internal variability samples
# -----------------------------------------------------------------------------
def control_realisations(
    control: xr.DataArray,
    region: str,
    n_blocks: int,
    block_years: int,
) -> np.ndarray:
    """Construct independent y-like piControl realizations.

    Each model/member piControl series is first averaged into non-overlapping ``block_years`` means.
    Those block means are then divided into independent, non-overlapping segments of length ``n_blocks``.
    """
    series = region_total(control, region)
    leading = [d for d in series.dims if d != "time"]
    if leading:
        series = series.stack(realisation=leading)
    else:
        series = series.expand_dims(realisation=[0])

    rows = []
    for i in range(series.sizes["realisation"]):
        v = np.asarray(series.isel(realisation=i).values, float)
        v = v[np.isfinite(v)]

        n_annual = (len(v) // block_years) * block_years
        if n_annual < n_blocks * block_years:
            continue

        blocks = v[:n_annual].reshape(-1, block_years).mean(axis=1)
        n_segments = len(blocks) // n_blocks

        for j in range(n_segments):
            rows.append(blocks[j * n_blocks : (j + 1) * n_blocks])

    if len(rows) < 6:
        raise ValueError(
            f"Only {len(rows)} independent piControl realizations were created "
            f"for {region}; more control data are required."
        )

    return np.asarray(rows, dtype=float)


def split_control(
    rows: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split independent control realizations into covariance (Z1) and test (Z2)."""
    n = len(rows)
    if method == "alternating":
        z1, z2 = rows[::2], rows[1::2]
    elif method == "segment":
        half = n // 2
        z1, z2 = rows[half:], rows[:half]
    elif method == "random":
        idx = np.random.default_rng(seed).permutation(n)
        half = n // 2
        z1, z2 = rows[idx[half:]], rows[idx[:half]]
    else:
        raise ValueError(method)

    if len(z1) < 2 or len(z2) < 2:
        raise ValueError("Both Z1 and Z2 require at least two realizations.")
    return z1, z2


# -----------------------------------------------------------------------------
# Effective ensemble size supplied to PyDnA
# -----------------------------------------------------------------------------
def member_counts_by_model(
    forced: xr.DataArray,
    forcing: str,
) -> pd.Series:
    """Number of available members per model for a forcing experiment."""
    da = forced.sel(forcing=forcing)

    if "model" not in da.dims:
        if "member" not in da.dims:
            return pd.Series({"single_model": 1.0})

        other = [d for d in da.dims if d != "member"]
        valid = da.notnull().any(other)
        return pd.Series({"single_model": float(valid.sum().item())})

    models = [str(v) for v in da.model.values]

    if "member" not in da.dims:
        return pd.Series(1.0, index=models)

    other = [d for d in da.dims if d not in ("model", "member")]
    valid = da.notnull().any(other)
    count = valid.sum("member").values.astype(float)
    return pd.Series(count, index=models)


def effective_n_direct(forced: xr.DataArray, forcing: str) -> float:
    """Equivalent n for an equal-weight mean of model ensemble means.

    If model m has n_m members and all members share the same internal-noise covariance, Var(MME) = C/M^2 * sum(1/n_m).
    The equivalent ensemble size is therefore M^2 / sum(1/n_m).
    """
    n = member_counts_by_model(forced, forcing)
    n = n[n > 0]
    m = len(n)
    return float(m * m / np.sum(1.0 / n.values))


def effective_n_ant(forced: xr.DataArray) -> float:
    """Equivalent n for ANT = ALL - NAT, assuming independent ALL/NAT noise."""
    n_all = member_counts_by_model(forced, "ALL")
    n_nat = member_counts_by_model(forced, "NAT")
    common = n_all.index.intersection(n_nat.index)

    a = n_all.loc[common]
    n = n_nat.loc[common]
    ok = (a > 0) & (n > 0)
    a, n = a[ok], n[ok]

    m = len(a)
    return float(m * m / np.sum(1.0 / a.values + 1.0 / n.values))


def effective_ensemble_sizes(forced: xr.DataArray) -> dict[str, float]:
    out = {f: effective_n_direct(forced, f) for f in FORCINGS}
    out["ANT"] = effective_n_ant(forced)
    return out


# -----------------------------------------------------------------------------
# Regularized TLS and residual-consistency test
# -----------------------------------------------------------------------------
def run_pydna_tls(
    y: np.ndarray,
    X: np.ndarray,
    control_rows: np.ndarray,
    nx: np.ndarray,
    pda,
    tls_formula: str,
    consistency_method: str,
    n_consistency: int,
    split_method: str,
    seed: int,
) -> dict:
    """Run regularized TLS and residual-consistency test."""
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]

    if len(y) != X.shape[0]:
        raise ValueError("y and X must have the same temporal dimension.")
    if X.shape[1] != len(nx):
        raise ValueError("One effective ensemble size is required per fingerprint.")

    # PyDnA/ROF removes one temporal degree of freedom by projection rather than by manually demeaning y, X, and control samples.
    U = np.asarray(pda.projfullrank(len(y), 1), dtype=float)
    yc = U @ y[:, None]
    Xc = U @ X

    z1_rows, z2_rows = split_control(control_rows, split_method, seed)
    Z1c = U @ z1_rows.T
    Z2c = U @ z2_rows.T

    # PyDnA.regC is the regularized covariance estimate used for prewhitening.
    Cf = np.asarray(pda.regC(np.matrix(Z1c.T)), dtype=float)
    Cf_sqrt = real_matrix(spla.sqrtm(Cf), "sqrt(Cf)")
    Cf_inv_sqrt = real_matrix(spla.inv(Cf_sqrt), "inv(sqrt(Cf))")

    # PyDnA.tls follows the row-oriented convention used in ROF_main.py:
    # X: n_signal x n_dimension, Y: 1 x n_dimension, Z2: n2 x n_dimension.
    Xw = np.matrix(Xc.T @ Cf_inv_sqrt)
    Yw = np.matrix(yc.T @ Cf_inv_sqrt)
    Z2w = np.matrix(Z2c.T @ Cf_inv_sqrt)
    nx_matrix = np.matrix(np.asarray(nx, float).reshape(1, -1))
    projection = np.identity(X.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        with legacy_numpy_seed(seed), pydna_tls_numpy_compat(pda):
            beta, lower, upper, d_cons, _, _ = pda.tls(
                Xw,
                Yw,
                Z2w,
                nx_matrix,
                projection,
                tls_formula,
            )

    beta = np.asarray(beta, float).reshape(-1)
    lower = np.asarray(lower, float).reshape(-1)
    upper = np.asarray(upper, float).reshape(-1)
    d_cons = float(np.asarray(d_cons).squeeze())

    # A lower limit greater than the upper limit is PyDnA's representation of an interval that crosses infinity.
    # Keep that information explicit rather than swapping limits.
    ci_unbounded = (~np.isfinite(lower)) | (~np.isfinite(upper)) | (lower > upper)

    n_red = Xc.shape[0]
    n_signal = Xc.shape[1]
    n1, n2 = len(z1_rows), len(z2_rows)

    if consistency_method == "MC":
        # Same Monte-Carlo residual-consistency procedure used by ROF_main.da(..., reg='TLS', cons_test='MC').
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            with legacy_numpy_seed(seed + 1):
                d_h0 = pda.consist_mc_tls(
                    np.matrix(Cf),
                    np.matrix(Xc),
                    nx_matrix,
                    n1,
                    n2,
                    n_consistency,
                    tls_formula,
                )

        d_h0 = np.asarray(d_h0, float).reshape(-1)
        p_cons = float(np.asarray(pda.gke(d_h0, d_cons)).squeeze())
        p_empirical = float(np.mean(d_h0 >= d_cons))
    else:
        # Allen & Stott (2003) parametric approximation, as implemented in PyDnA/ROF_main.py.
        df1 = n_red - n_signal
        if df1 <= 0:
            p_cons = np.nan
        else:
            p_cons = float(1.0 - sps.f.cdf(d_cons / df1, df1, n2))
        p_empirical = np.nan
        d_h0 = np.array([], dtype=float)

    return {
        "beta": beta,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_unbounded": ci_unbounded,
        "d_cons": d_cons,
        "p_consistency": p_cons,
        "p_consistency_empirical": p_empirical,
        "n_z1": n1,
        "n_z2": n2,
        "n_red": n_red,
        "n_signal": n_signal,
    }


# -----------------------------------------------------------------------------
# Trend maps and attributable trends
# -----------------------------------------------------------------------------
def slope_decade(da: xr.DataArray) -> xr.DataArray:
    x = time_years(da).astype(float)
    x = x - x.mean()

    def slope(y):
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return np.nan
        return sps.linregress(x[ok], y[ok]).slope * 10.0

    return xr.apply_ufunc(
        slope,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def calculate_mme_maps(forced: xr.DataArray) -> xr.Dataset:
    out = {}
    for forcing in ("ALL", "GHG", "AER"):
        per_model = slope_decade(model_mean(forced.sel(forcing=forcing)))
        per_model = ensure_model_dim(per_model)
        mme = per_model.mean("model", skipna=True)

        out[f"{forcing}_trend"] = mme
        out[f"{forcing}_agreement"] = (
            np.sign(per_model) == np.sign(mme)
        ).sum("model")

    return xr.Dataset(out)


def model_attributable_trends(
    forced: xr.DataArray,
    region: str,
    forcing: str,
    beta: float,
) -> xr.DataArray:
    """Per-model trend multiplied by the one-signal scaling factor."""
    if forcing == "ANT":
        da = ant_model_series(forced, region)
    else:
        da = forcing_model_series(forced, forcing, region)
    return slope_decade(da) * beta


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    obs = select_period(load_var(args.observed, args.var), args.start_year, args.end_year)
    forced = select_period(load_var(args.forced, args.var), args.start_year, args.end_year)
    control = load_var(args.control, args.var)

    available = {str(v) for v in forced.forcing.values}
    missing = [f for f in FORCINGS if f not in available]
    if missing:
        raise ValueError(f"Missing forcings in --forced: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    calculate_mme_maps(forced).to_netcdf(args.output_dir / "mme_trends.nc")

    nx_all = effective_ensemble_sizes(forced)
    pd.DataFrame(
        {"forcing": list(nx_all), "effective_ensemble_size": list(nx_all.values())}
    ).to_csv(args.output_dir / "fingerprint_ensemble_sizes.csv", index=False)

    analysis_groups = {
        "1-signal": [
            ("ALL",),
            ("NAT",),
            ("ANT",),
            ("GHG",),
            ("AER",),
        ],
        "2-signal": [
            ("NAT", "ANT"),
        ],
        "3-signal": [
            ("NAT", "GHG", "AER"),
        ],
    }

    scaling_rows = []
    consistency_rows = []
    attributable_rows = []
    block_metadata = {}

    for r_index, region in enumerate(REGIONS):
        obs_region = region_total(obs, region)
        signals = mme_series(forced, region)

        y, blocked_signals, labels = make_block_fingerprints(
            obs_region,
            signals,
            args.start_year,
            args.end_year,
            args.block_years,
        )
        block_metadata[region] = labels

        control_rows = control_realisations(
            control,
            region,
            n_blocks=len(y),
            block_years=args.block_years,
        )

        one_signal_beta = {}

        regression_index = 0
        for analysis, groups in analysis_groups.items():
            for group in groups:
                X = np.column_stack([blocked_signals[f] for f in group])
                nx = np.asarray([nx_all[f] for f in group], dtype=float)

                result = run_pydna_tls(
                    y=y,
                    X=X,
                    control_rows=control_rows,
                    nx=nx,
                    pda=pda,
                    tls_formula=args.tls_ci,
                    consistency_method=args.consistency,
                    n_consistency=args.n_consistency,
                    split_method=args.control_split,
                    seed=args.seed + 1000 * r_index + regression_index,
                )
                regression_index += 1

                signal_label = "+".join(group)
                consistency_rows.append(
                    {
                        "region": region,
                        "analysis": analysis,
                        "signals": signal_label,
                        "statistic": result["d_cons"],
                        "p_value": result["p_consistency"],
                        "empirical_p_value": result["p_consistency_empirical"],
                        "method": args.consistency,
                        "tls_ci_formula": args.tls_ci,
                        "n_Z1": result["n_z1"],
                        "n_Z2": result["n_z2"],
                        "n_reduced_dimensions": result["n_red"],
                    }
                )

                for i, forcing in enumerate(group):
                    beta = result["beta"][i]
                    lower = result["ci_lower"][i]
                    upper = result["ci_upper"][i]
                    unbounded = bool(result["ci_unbounded"][i])

                    scaling_rows.append(
                        {
                            # Keep these first six columns compatible with 04_plot_fig4.py.
                            "region": region,
                            "analysis": analysis,
                            "forcing": forcing,
                            "beta": beta,
                            "ci_lower": lower,
                            "ci_upper": upper,

                            # Additional diagnostics.
                            "ci_unbounded": unbounded,
                            "detected": (
                                bool(np.isfinite(lower) and lower > 0)
                                if not unbounded
                                else False
                            ),
                            "effective_ensemble_size": nx[i],
                            "consistency_p": result["p_consistency"],
                        }
                    )

                    if analysis == "1-signal":
                        one_signal_beta[forcing] = beta

        # Observed trends
        obs_year = time_years(obs_region).astype(float)
        obs_values = np.asarray(obs_region.values, float)
        observed_trend = sps.linregress(obs_year, obs_values).slope * 10.0
        attributable_rows.append(
            {
                "region": region,
                "forcing": "OBS",
                "sample": "OBS",
                "trend": observed_trend,
            }
        )

        # Per-model one-signal attributable trends
        for forcing in ATTRIBUTABLE_FORCINGS:
            values = model_attributable_trends(
                forced,
                region,
                forcing,
                one_signal_beta[forcing],
            )
            model_names = (
                [str(v) for v in values.model.values]
                if "model" in values.dims
                else ["single_model"]
            )

            for model_name, value in zip(model_names, np.asarray(values.values).reshape(-1)):
                if np.isfinite(value):
                    attributable_rows.append(
                        {
                            "region": region,
                            "forcing": forcing,
                            "sample": model_name,
                            "trend": float(value),
                        }
                    )

    pd.DataFrame(scaling_rows).to_csv(
        args.output_dir / "scaling_factors.csv", index=False
    )
    pd.DataFrame(consistency_rows).to_csv(
        args.output_dir / "residual_consistency.csv", index=False
    )
    pd.DataFrame(attributable_rows).to_csv(
        args.output_dir / "attributable_trends.csv", index=False
    )

    metadata = {
        "analysis_period": [args.start_year, args.end_year],
        "block_years": args.block_years,
        "blocks_used": block_metadata,
        "tls": "PyDnA.tls",
        "tls_ci_formula": args.tls_ci,
        "covariance": "PyDnA.regC",
        "residual_consistency": (
            "PyDnA.consist_mc_tls + PyDnA.gke"
            if args.consistency == "MC"
            else "Allen & Stott (2003) parametric F approximation"
        ),
        "control_split": args.control_split,
        "n_consistency": args.n_consistency,
        "random_seed": args.seed,
    }
    with open(args.output_dir / "attribution_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nPyDnA optimal fingerprinting completed.")
    print(f"  TLS CI formula: {args.tls_ci}")
    print(f"  Residual consistency: {args.consistency}")
    print(f"  Control split: {args.control_split}")
    print(f"  Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()