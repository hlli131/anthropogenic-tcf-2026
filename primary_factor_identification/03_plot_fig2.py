#!/usr/bin/env python3
"""Generate Fig.2 from processed data.

Panels
----------
a–c  Model performance for RF, XGBoost, and LightGBM.
d–e  SHAP values and mean absolute SHAP importance for XGBoost.
f–h  PDP/ICE responses for TCHP, W500, and MLD.
i–k  One-dimensional ALE responses for the same predictors.
l–m  Two-dimensional partial-dependence interactions.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cmaps
import joblib
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


FEATURES = (
    "RH600", "RV850", "AV850", "VWS", "W500", "SST",
    "SSS", "MLD", "D26", "T100", "TCHP", "MPI",
)
FEATURE_COLUMNS = tuple(f"ALL_{name}" for name in FEATURES)
TARGET = "All"
SPLIT_SEED = 1577

ATMOSPHERIC = {"RH600", "W500", "VWS", "RV850", "AV850"}
OCEANIC = {"SST", "SSS", "MLD", "D26", "TCHP", "T100"}

MODEL_FILES = {
    "RF": "rf.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot manuscript Fig. 2.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Fig2.pdf",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ale-bootstrap", type=int, default=1000)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 22,
        }
    )


def regression_ci(ax, x, y, color, alpha=0.25) -> None:
    result = stats.linregress(x, y)
    x_new = np.linspace(np.min(x), np.max(x), 100)
    y_fit = result.intercept + result.slope * x_new

    n = len(x)
    residual = y - (result.intercept + result.slope * x)
    mse = np.sum(residual**2) / (n - 2)
    tcrit = stats.t.ppf(0.975, n - 2)
    ci = tcrit * np.sqrt(mse) * np.sqrt(
        1 / n + (x_new - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    )

    ax.plot(x_new, y_fit, color=color, lw=2)
    ax.fill_between(x_new, y_fit - ci, y_fit + ci, color=color, alpha=alpha)


def plot_performance(ax, model, X_train, X_test, y_train, y_test, title, panel) -> None:
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_color = "#0d7ca1"
    test_color = "#ec721b"

    ax.scatter(y_train, pred_train, s=65, c=train_color, alpha=0.8, ec="k", lw=0.8)
    ax.scatter(y_test, pred_test, s=65, c=test_color, alpha=0.8, ec="k", lw=0.8)

    lo = min(y_train.min(), y_test.min(), pred_train.min(), pred_test.min())
    hi = max(y_train.max(), y_test.max(), pred_train.max(), pred_test.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)

    regression_ci(ax, y_train.to_numpy(), pred_train, train_color)
    regression_ci(ax, y_test.to_numpy(), pred_test, test_color)

    train_r2 = r2_score(y_train, pred_train)
    train_rmse = root_mean_squared_error(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)
    test_rmse = root_mean_squared_error(y_test, pred_test)

    ax.text(
        0.03, 0.97,
        f"Training\n$R^2$={train_r2:.2f}\nRMSE={train_rmse:.2f}",
        color=train_color, transform=ax.transAxes, va="top", fontsize=20,
    )
    ax.text(
        0.97, 0.03,
        f"Test\n$R^2$={test_r2:.2f}\nRMSE={test_rmse:.2f}",
        color=test_color, transform=ax.transAxes, ha="right", va="bottom", fontsize=20,
    )

    ax.set_xlabel("Observed TCF")
    if panel == "a":
        ax.set_ylabel("Predicted TCF")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def get_pd_ice(model, X, feature, grid_resolution=8):
    result = partial_dependence(
        model,
        X,
        features=[feature],
        kind="both",
        grid_resolution=grid_resolution,
    )
    return (
        np.asarray(result["grid_values"][0]),
        np.asarray(result["average"][0]),
        np.asarray(result["individual"][0]),
    )


def plot_pd_ice(ax, model, X, feature, label, panel) -> None:
    grid, average, individual = get_pd_ice(model, X, feature)

    for line in individual:
        ax.plot(grid, line, color="0.65", lw=0.7, alpha=0.7)
    ax.plot(grid, average, color="k", lw=2.2)

    hist_ax = ax.twinx()
    hist_ax.hist(X[feature], bins=10, color="C0", alpha=0.15, edgecolor="white")
    hist_ax.set_yticks([])
    hist_ax.set_ylabel("Frequency" if panel == "h" else "")

    ax.set_xlabel(label)
    if panel == "f":
        ax.set_ylabel("TCF (PDP & ICE)")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left")
    ax.set_zorder(hist_ax.get_zorder() + 1)
    ax.patch.set_visible(False)


def ale_1d(model, X, feature, n_bins=8):
    values = X[feature].to_numpy(float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"Not enough unique values for ALE: {feature}")

    bin_id = np.digitize(values, edges[1:-1], right=True)
    effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for i in range(len(effects)):
        mask = bin_id == i
        counts[i] = mask.sum()
        if not counts[i]:
            continue

        low = X.loc[mask].copy()
        high = low.copy()
        low[feature] = edges[i]
        high[feature] = edges[i + 1]
        effects[i] = np.mean(model.predict(high) - model.predict(low))

    cumulative = np.cumsum(effects)
    cumulative -= np.sum(cumulative * counts / counts.sum())
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, cumulative


def bootstrap_ale(model, X, feature, n_bootstrap=1000, n_bins=8, seed=1577):
    centers, effect = ale_1d(model, X, feature, n_bins=n_bins)
    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(X), len(X))
        sample = X.iloc[idx].reset_index(drop=True)
        try:
            x_b, y_b = ale_1d(model, sample, feature, n_bins=n_bins)
            curves.append(np.interp(centers, x_b, y_b))
        except ValueError:
            continue

    if not curves:
        return centers, effect, effect, effect

    curves = np.vstack(curves)
    return (
        centers,
        effect,
        np.percentile(curves, 2.5, axis=0),
        np.percentile(curves, 97.5, axis=0),
    )


def plot_ale(ax, model, X, feature, label, panel, n_bootstrap) -> None:
    x, effect, lower, upper = bootstrap_ale(
        model, X, feature, n_bootstrap=n_bootstrap
    )
    ax.plot(x, effect, color="r", lw=2.2)
    ax.fill_between(x, lower, upper, color="r", alpha=0.25)
    ax.axhline(0, color="0.5", ls="--", lw=1.2)

    hist_ax = ax.twinx()
    hist_ax.hist(X[feature], bins=10, color="C0", alpha=0.15, edgecolor="white")
    hist_ax.set_yticks([])
    hist_ax.set_ylabel("Frequency" if panel == "k" else "")

    ax.set_xlabel(label)
    if panel == "i":
        ax.set_ylabel(r"$\Delta$TCF (ALE)")
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left")
    ax.set_zorder(hist_ax.get_zorder() + 1)
    ax.patch.set_visible(False)


def interaction_pd(model, X, y_feature, x_feature, grid_resolution=20):
    yi = X.columns.get_loc(y_feature)
    xi = X.columns.get_loc(x_feature)
    result = partial_dependence(
        model,
        X,
        features=[(yi, xi)],
        kind="average",
        grid_resolution=grid_resolution,
    )
    y_grid, x_grid = result["grid_values"]
    surface = np.asarray(result["average"][0]) - np.mean(model.predict(X))
    return np.asarray(x_grid), np.asarray(y_grid), surface


def plot_interaction(
    fig, ax, model, X, y_feature, x_feature,
    x_label, y_label, panel, scale_y=1.0, add_colorbar=False,
) -> None:
    x, y, z = interaction_pd(model, X, y_feature, x_feature)
    y = y * scale_y

    cf = ax.contourf(
        x, y, z,
        levels=np.linspace(-2, 2, 41),
        cmap=cmaps.WhiteBlueGreenYellowRed,
        extend="both",
    )
    contours = ax.contour(
        x, y, z,
        levels=np.arange(-1.5, 1.6, 0.5),
        colors="k",
        linewidths=0.8,
    )
    ax.clabel(contours, fontsize=20, fmt="%.1f")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(rf"$\mathbf{{{panel}}}$", loc="left")

    if add_colorbar:
        cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.03)
        cbar.set_label(r"$\Delta$TCF")


def feature_category_color(name: str) -> str:
    if name in ATMOSPHERIC:
        return "#e43d33"
    if name in OCEANIC:
        return "#4977ba"
    return "#f5ba03"


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    data = pd.read_csv(args.data_dir / "IML_data.csv")
    X = data.loc[:, FEATURE_COLUMNS].astype(float)
    y = data[TARGET].astype(float)

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.3, random_state=SPLIT_SEED
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_dir = args.data_dir / "models"
    models = {
        name: joblib.load(model_dir / filename)
        for name, filename in MODEL_FILES.items()
    }
    xgb = models["XGBoost"]

    fig = plt.figure(figsize=(20, 13), dpi=args.dpi)
    gs = gridspec.GridSpec(3, 5, figure=fig, wspace=0.45, hspace=0.45)

    # a–c: predictive performance
    for col, (panel, name) in enumerate(zip("abc", MODEL_FILES)):
        plot_performance(
            fig.add_subplot(gs[0, col]),
            models[name], X_train, X_test, y_train, y_test, name, panel,
        )

    # d: SHAP beeswarm
    ax_d = fig.add_subplot(gs[0:2, 3])
    shap_values = shap.Explainer(xgb)(X)
    shap.plots.beeswarm(
        shap_values,
        max_display=len(FEATURES),
        show=False,
        color=cmaps.sunshine_diff_12lev,
        plot_size=None,
        ax=ax_d,
    )
    ax_d.set_title(r"$\mathbf{d}$", loc="left")
    ax_d.set_title("SHAP value")
    ax_d.set_xlabel("SHAP value")
    ax_d.set_yticklabels([t.get_text().replace("ALL_", "") for t in ax_d.get_yticklabels()])

    # e: mean absolute SHAP importance
    ax_e = fig.add_subplot(gs[0:2, 4])
    importance = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=[c.replace("ALL_", "") for c in FEATURE_COLUMNS],
    ).sort_values()
    colors = [feature_category_color(name) for name in importance.index]
    ax_e.barh(importance.index, importance.values, color=colors, alpha=0.8, edgecolor="k")
    ax_e.set_title(r"$\mathbf{e}$", loc="left")
    ax_e.set_title("Feature importance")
    ax_e.set_xlabel("Mean |SHAP value|")
    ax_e.legend(
        handles=[
            mpatches.Patch(fc="#e43d33", ec="k", label="Atmospheric"),
            mpatches.Patch(fc="#4977ba", ec="k", label="Oceanic"),
            mpatches.Patch(fc="#f5ba03", ec="k", label="Integrated"),
        ],
        frameon=False,
        fontsize=18,
        loc="lower right",
    )

    response_features = [
        ("ALL_TCHP", r"TCHP (kJ cm$^{-2}$)"),
        ("ALL_W500", r"W500 (Pa s$^{-1}$)"),
        ("ALL_MLD", "MLD (m)"),
    ]

    # f–h: PDP and ICE
    for col, ((feature, label), panel) in enumerate(zip(response_features, "fgh")):
        plot_pd_ice(fig.add_subplot(gs[1, col]), xgb, X, feature, label, panel)

    # i–k: ALE
    for col, ((feature, label), panel) in enumerate(zip(response_features, "ijk")):
        plot_ale(
            fig.add_subplot(gs[2, col]),
            xgb, X, feature, label, panel, args.ale_bootstrap,
        )

    # l–m: pairwise interactions
    plot_interaction(
        fig,
        fig.add_subplot(gs[2, 3]),
        xgb, X,
        y_feature="ALL_RV850",
        x_feature="ALL_VWS",
        x_label=r"VWS (m s$^{-1}$)",
        y_label=r"RV850 ($10^{-6}$ s$^{-1}$)",
        panel="l",
        scale_y=1e6,
    )
    plot_interaction(
        fig,
        fig.add_subplot(gs[2, 4]),
        xgb, X,
        y_feature="ALL_RH600",
        x_feature="ALL_SSS",
        x_label="SSS (PSU)",
        y_label="RH600 (%)",
        panel="m",
        add_colorbar=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Figure written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()