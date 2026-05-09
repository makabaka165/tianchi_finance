from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--run-name", default="lgb_baseline", help="Prefix used for generated file names")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=2000, help="Maximum boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=64, help="Number of leaves")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds")
    parser.add_argument("--sample-rows", type=int, default=0, help="Use first N train rows for a quick smoke test")
    parser.add_argument("--n-jobs", type=int, default=-1, help="LightGBM thread count")
    parser.add_argument("--target-encoding", action="store_true", help="Add fold-safe target encoding features")
    parser.add_argument("--target-encoding-splits", type=int, default=5, help="Inner folds for train target encoding")
    parser.add_argument("--target-encoding-smoothing", type=float, default=20.0, help="Smoothing for target encoding")
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
    df["issue_quarter"] = issue_date.dt.quarter.astype("float32")
    df["issue_half_year"] = ((df["issue_month"] > 6).astype("float32") + 1).astype("float32")
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
    df["credit_history_years"] = (df["credit_history_months"] / 12).astype("float32")
    return df


def add_grade_features(df: pd.DataFrame) -> pd.DataFrame:
    grade_map = {grade: idx for idx, grade in enumerate("ABCDEFG", start=1)}
    grade_rank = df["grade"].map(grade_map).astype("float32")
    subgrade = df["subGrade"].astype("string").str.extract(r"([A-G])(\d+)")
    subgrade_grade = subgrade[0].map(grade_map).astype("float32")
    subgrade_number = pd.to_numeric(subgrade[1], errors="coerce").astype("float32")

    df["grade_rank"] = grade_rank
    df["subgrade_rank"] = ((subgrade_grade - 1) * 5 + subgrade_number).astype("float32")
    return df


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    annual_income = df["annualIncome"].replace(0, np.nan)
    total_acc = df["totalAcc"].replace(0, np.nan)
    open_acc = df["openAcc"].replace(0, np.nan)
    term_months = (df["term"] * 12).replace(0, np.nan)

    df["fico_mean"] = ((df["ficoRangeLow"] + df["ficoRangeHigh"]) / 2).astype("float32")
    df["fico_range"] = (df["ficoRangeHigh"] - df["ficoRangeLow"]).astype("float32")
    df["term_months"] = term_months.astype("float32")
    df["loan_income_ratio"] = (df["loanAmnt"] / annual_income).astype("float32")
    df["installment_income_ratio"] = ((df["installment"] * 12) / annual_income).astype("float32")
    df["revolbal_income_ratio"] = (df["revolBal"] / annual_income).astype("float32")
    df["openacc_totalacc_ratio"] = (open_acc / total_acc).astype("float32")
    df["revolbal_openacc_ratio"] = (df["revolBal"] / open_acc).astype("float32")
    df["loan_term_ratio"] = (df["loanAmnt"] / term_months).astype("float32")
    df["interest_loan_income_ratio"] = (df["interestRate"] * df["loan_income_ratio"]).astype("float32")
    df["dti_loan_income_ratio"] = (df["dti"] * df["loan_income_ratio"]).astype("float32")
    df["dti_installment_income_ratio"] = (df["dti"] * df["installment_income_ratio"]).astype("float32")
    df["revolutil_dti_ratio"] = (df["revolUtil"] * df["dti"]).astype("float32")
    df["annual_income_log1p"] = np.log1p(df["annualIncome"].clip(lower=0)).astype("float32")
    df["loan_amnt_log1p"] = np.log1p(df["loanAmnt"].clip(lower=0)).astype("float32")
    df["installment_log1p"] = np.log1p(df["installment"].clip(lower=0)).astype("float32")
    df["revolbal_log1p"] = np.log1p(df["revolBal"].clip(lower=0)).astype("float32")
    return df


def add_anonymous_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    n_cols = [f"n{i}" for i in range(15) if f"n{i}" in df.columns]
    if not n_cols:
        return df

    n_data = df[n_cols]
    df["n_missing_count"] = n_data.isna().sum(axis=1).astype("float32")
    df["n_zero_count"] = n_data.eq(0).sum(axis=1).astype("float32")
    df["n_sum"] = n_data.sum(axis=1, skipna=True).astype("float32")
    df["n_mean"] = n_data.mean(axis=1, skipna=True).astype("float32")
    df["n_std"] = n_data.std(axis=1, skipna=True).astype("float32")
    df["n_max"] = n_data.max(axis=1, skipna=True).astype("float32")
    df["n_min"] = n_data.min(axis=1, skipna=True).astype("float32")
    return df


def add_count_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        key = df[col].astype("string").fillna("__MISSING__")
        counts = key.value_counts(dropna=False)
        df[f"{col}_count"] = key.map(counts).astype("float32")
    return df


def add_group_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["postCode", "regionCode", "purpose", "grade", "subGrade", "homeOwnership"]
    agg_cols = ["loanAmnt", "annualIncome", "interestRate", "dti", "revolUtil", "installment"]
    new_features: dict[str, pd.Series] = {}

    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        group_key = df[group_col].astype("string").fillna("__MISSING__")
        for agg_col in agg_cols:
            if agg_col not in df.columns:
                continue
            group_mean = df[agg_col].groupby(group_key, dropna=False).transform("mean")
            new_features[f"{agg_col}_mean_by_{group_col}"] = group_mean.astype("float32")
            new_features[f"{agg_col}_diff_mean_by_{group_col}"] = (
                df[agg_col] - group_mean
            ).astype("float32")

    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
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


def make_category_key(series: pd.Series) -> pd.Series:
    return series.fillna(-999999).astype("float64")


def fit_smoothed_target_mapping(
    keys: pd.Series,
    target: np.ndarray,
    global_mean: float,
    smoothing: float,
) -> pd.Series:
    stats = pd.DataFrame({"key": keys, "target": target}).groupby("key", dropna=False)["target"].agg(
        ["mean", "count"]
    )
    return (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)


def map_target_encoding(keys: pd.Series, mapping: pd.Series, global_mean: float) -> pd.Series:
    return keys.map(mapping).fillna(global_mean).astype("float32")


def add_fold_target_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list[str],
    inner_splits: int,
    seed: int,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = [col for col in cols if col in X_train.columns]
    if not target_cols:
        return X_train, X_valid, X_test

    y_train = np.asarray(y_train)
    global_mean = float(np.mean(y_train))
    min_class_count = int(np.bincount(y_train).min())
    inner_splits = max(2, min(inner_splits, min_class_count))

    train_features: dict[str, np.ndarray] = {}
    valid_features: dict[str, pd.Series] = {}
    test_features: dict[str, pd.Series] = {}
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    for col in target_cols:
        train_key = make_category_key(X_train[col]).reset_index(drop=True)
        valid_key = make_category_key(X_valid[col])
        test_key = make_category_key(X_test[col])

        train_encoded = np.full(len(X_train), global_mean, dtype=np.float32)
        for inner_train_idx, inner_valid_idx in inner_cv.split(np.zeros(len(y_train)), y_train):
            mapping = fit_smoothed_target_mapping(
                train_key.iloc[inner_train_idx],
                y_train[inner_train_idx],
                global_mean,
                smoothing,
            )
            train_encoded[inner_valid_idx] = map_target_encoding(
                train_key.iloc[inner_valid_idx],
                mapping,
                global_mean,
            ).to_numpy(dtype=np.float32)

        full_mapping = fit_smoothed_target_mapping(train_key, y_train, global_mean, smoothing)
        feature_name = f"{col}_target_mean"
        train_features[feature_name] = train_encoded
        valid_features[feature_name] = map_target_encoding(valid_key, full_mapping, global_mean)
        test_features[feature_name] = map_target_encoding(test_key, full_mapping, global_mean)

    X_train_aug = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(train_features)], axis=1)
    X_valid_aug = pd.concat([X_valid.reset_index(drop=True), pd.DataFrame(valid_features).reset_index(drop=True)], axis=1)
    X_test_aug = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(test_features).reset_index(drop=True)], axis=1)
    return X_train_aug, X_valid_aug, X_test_aug


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    y = train["isDefault"].astype("int8")
    train_x = train.drop(columns=["isDefault"])
    all_data = pd.concat([train_x, test], axis=0, ignore_index=True)

    all_data["employmentLength"] = employment_length_to_years(all_data["employmentLength"])
    all_data = add_date_features(all_data)
    all_data = add_grade_features(all_data)
    all_data = add_numeric_features(all_data)
    all_data = add_anonymous_aggregate_features(all_data)
    all_data = add_count_features(
        all_data,
        [
            "employmentTitle",
            "postCode",
            "title",
            "regionCode",
            "purpose",
            "grade",
            "subGrade",
            "homeOwnership",
            "verificationStatus",
        ],
    )
    all_data = add_group_stat_features(all_data)

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
    feature_importances: pd.DataFrame | None = None

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    target_encoding_cols = [
        "grade",
        "subGrade",
        "employmentTitle",
        "postCode",
        "regionCode",
        "purpose",
        "title",
        "homeOwnership",
        "verificationStatus",
    ]

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_array), start=1):
        print(f"\nFold {fold}/{args.n_splits}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_array[train_idx], y_array[valid_idx]
        X_test_fold = X_test

        if args.target_encoding:
            X_train, X_valid, X_test_fold = add_fold_target_encoding(
                X_train,
                y_train,
                X_valid,
                X_test,
                cols=target_encoding_cols,
                inner_splits=args.target_encoding_splits,
                seed=args.seed + fold,
                smoothing=args.target_encoding_smoothing,
            )

        if feature_importances is None:
            feature_importances = pd.DataFrame({"feature": X_train.columns})

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
            model.predict_proba(X_test_fold, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)
            / args.n_splits
        )
        feature_importances[f"fold_{fold}"] = model.feature_importances_
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y_array, oof)
    elapsed = time.time() - start

    submission = pd.DataFrame({"id": test_ids, "isDefault": test_pred})
    submission_path = output_dir / f"submission_{args.run_name}.csv"
    submission.to_csv(submission_path, index=False)

    oof_path = output_dir / f"oof_{args.run_name}.csv"
    pd.DataFrame({"isDefault": y_array, "pred": oof}).to_csv(oof_path, index=False)

    if feature_importances is None:
        raise RuntimeError("No model was trained, feature importances are unavailable.")
    feature_importances["importance_mean"] = feature_importances.filter(like="fold_").mean(axis=1)
    feature_importances = feature_importances.sort_values("importance_mean", ascending=False)
    importance_path = output_dir / f"feature_importance_{args.run_name}.csv"
    feature_importances.to_csv(importance_path, index=False)

    metrics = {
        "run_name": args.run_name,
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
        "target_encoding": {
            "enabled": args.target_encoding,
            "cols": target_encoding_cols if args.target_encoding else [],
            "inner_splits": args.target_encoding_splits if args.target_encoding else None,
            "smoothing": args.target_encoding_smoothing if args.target_encoding else None,
        },
    }
    metrics_path = output_dir / f"metrics_{args.run_name}.json"
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
