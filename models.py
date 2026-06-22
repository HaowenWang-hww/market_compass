"""Model training and prediction with purge/embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from features import FEATURE_COLS


LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split_with_purge(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    train_ratio: float = config.TRAIN_RATIO,
    embargo: int = config.MAX_H,
    backtest_start: str = config.BACKTEST_START,
) -> SplitResult:
    """70/30 temporal split with purge + embargo at boundary."""
    work = df[df["date"] >= pd.Timestamp(backtest_start)].copy()
    work = work.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    n = len(work)
    split_idx = int(n * train_ratio)

    train_raw = work.iloc[:split_idx]
    test_raw = work.iloc[split_idx:]

    train = train_raw.iloc[:-embargo].copy() if len(train_raw) > embargo else train_raw.iloc[:0]
    test = test_raw.iloc[embargo:].copy() if len(test_raw) > embargo else test_raw.iloc[:0]

    return SplitResult(
        train=train,
        test=test,
        train_end=train["date"].max() if len(train) else pd.NaT,
        test_start=test["date"].min() if len(test) else pd.NaT,
    )


def _trend_score_row(row: pd.Series, feature_cols: list[str]) -> float:
    f1, f2, f3, f4, f5, f6 = [row[c] for c in feature_cols[:6]]
    return (f1 + f2 + f6 + (1 - f3) + (1 - f4) + (1 - f5)) / 6.0


def equal_weight_predict_binary(X: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    feature_cols = feature_cols or FEATURE_COLS
    return np.array([
        1 if _trend_score_row(X.iloc[i], feature_cols) >= 0.45 else 0
        for i in range(len(X))
    ])


def equal_weight_predict_3state(
    X: pd.DataFrame,
    close: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    ma10 = close.rolling(10, min_periods=10).mean()
    ma30 = close.rolling(30, min_periods=30).mean()
    base_cols = feature_cols[:6]

    preds = []
    for i in range(len(X)):
        score = _trend_score_row(X.iloc[i], base_cols)
        if score < 0.45:
            preds.append(0)
        else:
            direction = 1 if ma10.iloc[i] > ma30.iloc[i] else -1
            if "mom_60" in X.columns and not pd.isna(X.iloc[i]["mom_60"]):
                direction = int(np.sign(X.iloc[i]["mom_60"])) or direction
            preds.append(direction)
    return np.array(preds)


def smooth_predictions(series: pd.Series, window: int = config.SMOOTH_WINDOW) -> pd.Series:
    def _mode(arr):
        vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
        return vals[counts.argmax()] if len(vals) else np.nan

    mapped = series.map(LABEL_TO_IDX)
    smoothed = mapped.rolling(window, min_periods=1).apply(_mode, raw=True)
    return smoothed.map(IDX_TO_LABEL)


def smooth_binary_predictions(series: pd.Series, window: int = config.SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(
        lambda x: int(np.round(np.nanmean(x))), raw=True
    ).astype(int)


@dataclass
class ModelResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_smooth: np.ndarray
    accuracy: float
    macro_f1: float
    recall_per_class: dict
    confusion: np.ndarray
    model: object | None = None
    task: str = "3state"


def _evaluate_binary(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    y_pred_s = smooth_binary_predictions(pd.Series(y_pred)).values
    labels = [0, 1]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return ModelResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_smooth=y_pred_s,
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        recall_per_class={lbl: recalls[i] for i, lbl in enumerate(labels)},
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        task="binary",
    )


def _evaluate_3state(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    y_pred_s = smooth_predictions(pd.Series(y_pred)).values
    labels = [-1, 0, 1]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return ModelResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_smooth=y_pred_s,
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        recall_per_class={
            lbl: recalls[i] for i, lbl in enumerate(labels)
        },
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        task="3state",
    )


def train_and_predict_binary(split: SplitResult, feature_cols: list[str] | None = None) -> dict:
    feature_cols = feature_cols or FEATURE_COLS
    train, test = split.train, split.test
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train["label_binary"].values, test["label_binary"].values
    weights = train["abs_t"].values if "abs_t" in train.columns else None

    eq_pred = equal_weight_predict_binary(X_test, feature_cols)
    eq_result = _evaluate_binary("等权模型", y_test, eq_pred)

    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight="balanced")
    lr.fit(X_train, y_train, sample_weight=weights)
    lr_pred = lr.predict(X_test)
    lr_result = _evaluate_binary("逻辑回归", y_test, lr_pred)
    lr_result.model = lr

    dt = DecisionTreeClassifier(
        max_depth=5, random_state=config.RANDOM_STATE, class_weight="balanced"
    )
    dt.fit(X_train, y_train, sample_weight=weights)
    dt_pred = dt.predict(X_test)
    dt_result = _evaluate_binary("决策树", y_test, dt_pred)
    dt_result.model = dt

    return {
        "equal_weight": eq_result,
        "logistic": lr_result,
        "decision_tree": dt_result,
        "test_df": test,
    }


def train_and_predict_3state(split: SplitResult, feature_cols: list[str]) -> dict:
    train, test = split.train, split.test
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train["label"].values, test["label"].values
    weights = train["abs_t"].values if "abs_t" in train.columns else None
    close_test = test.set_index("date")["close"]

    eq_pred = equal_weight_predict_3state(X_test, close_test, feature_cols)
    eq_result = _evaluate_3state("等权模型", y_test, eq_pred)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE, class_weight="balanced")),
    ])
    if weights is not None:
        lr.fit(X_train, y_train, clf__sample_weight=weights)
    else:
        lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_result = _evaluate_3state("逻辑回归", y_test, lr_pred)
    lr_result.model = lr

    dt = DecisionTreeClassifier(
        max_depth=6, random_state=config.RANDOM_STATE, class_weight="balanced"
    )
    dt.fit(X_train, y_train, sample_weight=weights)
    dt_pred = dt.predict(X_test)
    dt_result = _evaluate_3state("决策树", y_test, dt_pred)
    dt_result.model = dt

    return {
        "equal_weight": eq_result,
        "logistic": lr_result,
        "decision_tree": dt_result,
        "test_df": test,
    }


def predict_full_series_binary(
    df: pd.DataFrame,
    model_results: dict,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    feature_cols = feature_cols or FEATURE_COLS
    work = df.dropna(subset=feature_cols + ["label_binary"]).copy()
    X = work[feature_cols]

    out = work[["date", "close", "label_binary"]].copy()
    out["pred_eq"] = equal_weight_predict_binary(X, feature_cols)
    out["pred_lr"] = model_results["logistic"].model.predict(X)
    out["pred_dt"] = model_results["decision_tree"].model.predict(X)

    for col in ["pred_eq", "pred_lr", "pred_dt"]:
        out[f"{col}_smooth"] = smooth_binary_predictions(out[col]).values
        out[f"{col}_binary"] = out[col].astype(int)
        out[f"{col}_smooth_binary"] = out[f"{col}_smooth"].astype(int)

    return out


def predict_full_series_3state(
    df: pd.DataFrame,
    model_results: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    work = df.dropna(subset=feature_cols + ["label"]).copy()
    X = work[feature_cols]
    close = work.set_index("date")["close"]

    out = work[["date", "close", "label", "label_binary", "label_perfect", "abs_t"]].copy()
    out["pred_eq"] = equal_weight_predict_3state(X, close, feature_cols)
    out["pred_lr"] = model_results["logistic"].model.predict(X)
    out["pred_dt"] = model_results["decision_tree"].model.predict(X)

    for col in ["pred_eq", "pred_lr", "pred_dt"]:
        out[f"{col}_smooth"] = smooth_predictions(out[col]).values
        out[f"{col}_binary"] = (out[col] != 0).astype(int)
        out[f"{col}_smooth_binary"] = (out[f"{col}_smooth"] != 0).astype(int)

    return out
