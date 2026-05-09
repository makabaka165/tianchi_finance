from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM baseline for Tianchi finance default prediction.")
    parser.add_argument("--train-path", default="train.csv", help="Path to train.csv")
    parser.add_argument("--test-path", default="testA.csv", help="Path to testA.csv")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated files")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=2000, help="Maximum boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=64, help="Number of leaves")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds")
    parser.add_argument("--sample-rows", type=int, default=0, help="Use first N train rows for a quick smoke test")
    parser.add_argument("--n-jobs", type=int, default=-1, help="LightGBM thread count")
    return parser.parse_args()


def employment_length_to_years(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    years = text.str.extract(r"(\d+)", expand=False).astype("float32")
    years = years.mask(text.str.contains("< 1", na=False), 0)
    years = years.mask(text.str.contains("10+", regex=False, na=False), 10)
    return years


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    issue_date = pd.to_datetime(df["issueDate"], errors="coerce")
    df["issue_year"] = issue_date.dt.year.astype("float32")
    df["issue_month"] = issue_date.dt.month.astype("float32")
    df["issue_month_index"] = (df["issue_year"] * 12 + df["issue_month"]).astype("float32")

    early = df["earliesCreditLine"].astype("string").str.extract(r"([A-Za-z]{3})-(\d{4})")
    df["earlies_credit_month"] = early[0].map(MONTH_MAP).astype("float32")
    df["earlies_credit_year"] = pd.to_numeric(early[1], errors="coerce").astype("float32")
    df["earlies_credit_month_index"] = (
        df["earlies_credit_year"] * 12 + df["earlies_credit_month"]
    ).astype("float32")
    df["credit_history_months"] = (
        df["issue_month_index"] - df["earlies_credit_month_index"]
    ).astype("float32")
    return df


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    annual_income = df["annualIncome"].replace(0, np.nan)
    total_acc = df["totalAcc"].replace(0, np.nan)
    open_acc = df["openAcc"].replace(0, np.nan)

    df["fico_mean"] = ((df["ficoRangeLow"] + df["ficoRangeHigh"]) / 2).astype("float32")
    df["fico_range"] = (df["ficoRangeHigh"] - df["ficoRangeLow"]).astype("float32")
    df["loan_income_ratio"] = (df["loanAmnt"] / annual_income).astype("float32")
    df["installment_income_ratio"] = ((df["installment"] * 12) / annual_income).astype("float32")
    df["revolbal_income_ratio"] = (df["revolBal"] / annual_income).astype("float32")
    df["openacc_totalacc_ratio"] = (open_acc / total_acc).astype("float32")
    df["revolbal_openacc_ratio"] = (df["revolBal"] / open_acc).astype("float32")
    return df


def add_count_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        key = df[col].astype("string").fillna("__MISSING__")
        counts = key.value_counts(dropna=False)
        df[f"{col}_count"] = key.map(counts).astype("float32")
    return df


def label_encode_objects(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_cols:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes.astype("int32")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            if col == "id":
                continue
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    y = train["isDefault"].astype("int8")
    train_x = train.drop(columns=["isDefault"])
    all_data = pd.concat([train_x, test], axis=0, ignore_index=True)

    all_data["employmentLength"] = employment_length_to_years(all_data["employmentLength"])
    all_data = add_date_features(all_data)
    all_data = add_numeric_features(all_data)
    all_data = add_count_features(
        all_data,
        ["employmentTitle", "postCode", "title", "regionCode", "purpose"],
    )

    drop_cols = ["id", "issueDate", "earliesCreditLine", "policyCode"]
    all_data = all_data.drop(columns=[col for col in drop_cols if col in all_data.columns])
    all_data = label_encode_objects(all_data)
    all_data = reduce_memory(all_data)

    X = all_data.iloc[: len(train)].reset_index(drop=True)
    X_test = all_data.iloc[len(train) :].reset_index(drop=True)
    return X, X_test, y.tolist()


def train_and_predict(args: argparse.Namespace) -> dict[str, object]:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    if args.sample_rows and args.sample_rows > 0:
        train = train.head(args.sample_rows).copy()
        print(f"Using sample rows: {len(train)}")

    test_ids = test["id"].copy()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    print("Building features...")
    X, X_test, y = build_features(train, test)
    y_array = np.asarray(y)
    print(f"Feature shape: {X.shape}, Test feature shape: {X_test.shape}")

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbosity": -1,
        "force_col_wise": True,
    }

    oof = np.zeros(len(X), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_scores: list[float] = []
    feature_importances = pd.DataFrame({"feature": X.columns})

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_array), start=1):
        print(f"\nFold {fold}/{args.n_splits}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_array[train_idx], y_array[valid_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        valid_pred = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(float(fold_auc))
        oof[valid_idx] = valid_pred.astype(np.float32)

        test_pred += (
            model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)
            / args.n_splits
        )
        feature_importances[f"fold_{fold}"] = model.feature_importances_
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y_array, oof)
    elapsed = time.time() - start

    submission = pd.DataFrame({"id": test_ids, "isDefault": test_pred})
    submission_path = output_dir / "submission_lgb_baseline.csv"
    submission.to_csv(submission_path, index=False)

    oof_path = output_dir / "oof_lgb_baseline.csv"
    pd.DataFrame({"isDefault": y_array, "pred": oof}).to_csv(oof_path, index=False)

    feature_importances["importance_mean"] = feature_importances.filter(like="fold_").mean(axis=1)
    feature_importances = feature_importances.sort_values("importance_mean", ascending=False)
    importance_path = output_dir / "feature_importance_lgb_baseline.csv"
    feature_importances.to_csv(importance_path, index=False)

    metrics = {
        "cv_auc": float(cv_auc),
        "fold_auc": fold_scores,
        "elapsed_seconds": round(elapsed, 2),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": int(X.shape[1]),
        "submission_path": str(submission_path),
        "oof_path": str(oof_path),
        "importance_path": str(importance_path),
        "params": params,
    }
    metrics_path = output_dir / "metrics_lgb_baseline.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"CV AUC: {cv_auc:.6f}")
    print(f"Fold AUC: {[round(v, 6) for v in fold_scores]}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Submission: {submission_path}")
    print(f"OOF: {oof_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Metrics: {metrics_path}")
    return metrics


def main() -> None:
    args = parse_args()
    train_and_predict(args)


if __name__ == "__main__":
    main()
