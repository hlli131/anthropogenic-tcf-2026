#!/usr/bin/env python3
"""Train RF, XGBoost, and LightGBM models.

Outputs
----------
model_metrics.csv
model_predictions.csv
best_params.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor


FEATURES = (
    "RH600", "RV850", "AV850", "VWS", "W500", "SST",
    "SSS", "MLD", "D26", "T100", "TCHP", "MPI",
)
FEATURE_COLUMNS = tuple(f"ALL_{name}" for name in FEATURES)
TARGET = "All"
SPLIT_SEED = 1577


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ML models.")
    parser.add_argument("--data", type=Path, required=True, help="IML_data.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_data",
    )
    return parser.parse_args()


def model_specs():
    return {
        "RF": (
            RandomForestRegressor(
                n_estimators=30,
                criterion="friedman_mse",
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1.0,
                bootstrap=True,
                max_samples=1.0,
                n_jobs=-1,
                random_state=1983,
            ),
            {
                "n_estimators": range(5, 51),
                "max_depth": [2, 3, 4, 5],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2, 3],
                "max_samples": [0.8, 0.9, 1.0],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_child_weight=1.0,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=SPLIT_SEED,
                n_jobs=-1,
            ),
            {
                "n_estimators": range(5, 51),
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.1, 0.2, 0.3],
                "reg_alpha": [0.5, 1.0, 1.5],
                "reg_lambda": [0.4, 0.6, 0.8],
            },
        ),
        "LightGBM": (
            LGBMRegressor(
                num_leaves=10,
                max_depth=3,
                learning_rate=0.1,
                n_estimators=50,
                min_child_weight=1.0,
                min_child_samples=4,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_alpha=0.8,
                reg_lambda=0.8,
                random_state=SPLIT_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            {
                "num_leaves": [5, 10, 20, 30],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.05, 0.10, 0.15],
                "n_estimators": range(5, 51),
                "reg_alpha": [0.8, 1.0, 1.2],
                "reg_lambda": [0.2, 0.4, 0.6],
            },
        ),
    }


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)

    missing = [c for c in (*FEATURE_COLUMNS, TARGET) if c not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    valid = data.loc[:, [*FEATURE_COLUMNS, TARGET]].notna().all(axis=1)
    data = data.loc[valid].copy()

    train_idx, test_idx = train_test_split(
        data.index.to_numpy(),
        test_size=0.3,
        random_state=SPLIT_SEED,
    )
    X_train = data.loc[train_idx, FEATURE_COLUMNS]
    X_test = data.loc[test_idx, FEATURE_COLUMNS]
    y_train = data.loc[train_idx, TARGET]
    y_test = data.loc[test_idx, TARGET]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    metric_rows = []
    best_params = {}
    predictions = pd.DataFrame(
        {
            "row_index": data.index,
            "year": data["year"] if "year" in data else np.arange(len(data)),
            "observed": data[TARGET],
            "subset": np.where(data.index.isin(train_idx), "Training", "Test"),
        }
    ).set_index("row_index")

    for name, (estimator, grid) in model_specs().items():
        search = GridSearchCV(
            estimator,
            param_grid=grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

        joblib.dump(model, model_dir / f"{name.lower()}.joblib")
        best_params[name] = search.best_params_

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        predictions.loc[train_idx, name] = pred_train
        predictions.loc[test_idx, name] = pred_test

        for subset, truth, pred in (
            ("Training", y_train, pred_train),
            ("Test", y_test, pred_test),
        ):
            row = {"model": name, "subset": subset}
            row.update(calculate_metrics(truth, pred))
            metric_rows.append(row)

    pd.DataFrame(metric_rows).to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions.reset_index(drop=True).sort_values("year").to_csv(args.output_dir / "model_predictions.csv", index=False)
    with open(args.output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()