from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from baseline_lgb import (
    add_anonymous_aggregate_features,
    add_category_combo_features,
    add_count_features,
    add_date_features,
    add_grade_features,
    add_group_stat_features,
    add_numeric_features,
    employment_length_to_years,
)


BASE_CAT_COLS = [
    "grade",
    "subGrade",
    "employmentTitle",
    "homeOwnership",
    "verificationStatus",
    "purpose",
    "postCode",
    "regionCode",
    "title",
    "initialListStatus",
    "applicationType",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost baseline for Tianchi finance default prediction.")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--test-path", default="testA.csv")
    parser.add_argument("--output-dir", default="outputs_catboost")
    parser.add_argument("--run-name", default="cat_exp011")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--category-combos", action="store_true")
    parser.add_argument("--numeric-category-cols", action="store_true")
    parser.add_argument("--forum-features", action="store_true")
    return parser.parse_args()


FORUM_CAT_COLS = [
    "employmentLength_bin",
    "issueDate_bin",
    "earliesCreditLine_bin",
    "term_bin",
    "interestRate_bin",
    "annualIncome_bin",
    "loanAmnt_bin",
    "homeOwnership_bin",
    "dti_bin",
    "installment_bin",
    "revolBal_bin",
    "revolUtil_bin",
]


FORUM_RATIO_FEATURES = [
    "loanAmnt",
    "installment",
    "interestRate",
    "annualIncome",
    "dti",
    "openAcc",
    "revolBal",
    "revolUtil",
    "totalAcc",
]


FORUM_PSI_DROP_COLS = [
    "installment_homeOwnership_ratio",
    "installment_purpose_ratio",
    "revolBal_issueDate_ratio",
    "revolBal_loanAmnt",
    "annualIncome_installment",
    "installment_issueDate_ratio",
    "installment_employmentLength_ratio",
    "revolUtil_issueDate_ratio",
    "revolBal_purpose_ratio",
    "revolBal_homeOwnership_ratio",
    "revolBal_employmentLength_ratio",
    "dti_issueDate_ratio",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series | np.ndarray | float) -> pd.Series:
    denominator_series = pd.Series(denominator, index=numerator.index, dtype="float64").replace(0, np.nan)
    return (numerator.astype("float64") / denominator_series).replace([np.inf, -np.inf], np.nan).astype("float32")


def add_quantile_bin_feature(
    df: pd.DataFrame,
    source_col: str,
    feature_name: str,
    bins: int,
) -> str | None:
    if source_col not in df.columns:
        return None
    values = df[source_col].replace([np.inf, -np.inf], np.nan)
    try:
        binned = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    df[feature_name] = binned.fillna(-1).astype("int16")
    return feature_name


def add_group_median_ratio_features(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    feature_name: str,
) -> None:
    if value_col not in df.columns or group_col not in df.columns:
        return
    group_key = df[group_col].fillna(-999999)
    medians = df[value_col].groupby(group_key, dropna=False).transform("median")
    df[feature_name] = safe_ratio(df[value_col], medians)


def add_issue_date_window_features(df: pd.DataFrame, value_col: str) -> None:
    if value_col not in df.columns or "issue_month_index" not in df.columns:
        return
    issue_values = sorted(df["issue_month_index"].dropna().unique().tolist())
    value = df[value_col].astype("float64")
    issue = df["issue_month_index"]

    medians: dict[float, float] = {}
    for issue_value in issue_values:
        window_mask = issue.between(issue_value - 3, issue_value + 3)
        medians[issue_value] = float(value.loc[window_mask].median())
    median_feature = issue.map(medians).astype("float32")
    df[f"{value_col}_issueDate_median"] = median_feature
    df[f"{value_col}_issueDate_ratio"] = safe_ratio(df[value_col], median_feature)


def add_forum_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    forum_cat_cols = [col for col in FORUM_CAT_COLS if col in df.columns]

    df["date_Diff"] = (df["issue_month_index"] - df["earlies_credit_month_index"]).astype("float32")
    df["dti"] = df["dti"].abs().fillna(1000).astype("float32")
    df["installment_term_revolBal"] = safe_ratio(df["installment"] * 12 * df["term"], df["revolBal"] + 0.1)
    df["revolUtil_revolBal"] = safe_ratio(df["revolUtil"], df["revolBal"] + 0.1)
    df["openAcc_totalAcc"] = safe_ratio(df["openAcc"], df["totalAcc"])
    df["loanAmnt_dti_annualIncome"] = safe_ratio(df["loanAmnt"], df["dti"].abs() * df["annualIncome"] + 0.1)
    df["annualIncome_loanAmnt"] = safe_ratio(df["annualIncome"], df["loanAmnt"] + 0.1)
    df["revolBal_loanAmnt"] = safe_ratio(df["revolBal"], df["loanAmnt"] + 0.1)
    df["revolBal_installment"] = safe_ratio(df["revolBal"], df["installment"] + 0.1)
    df["annualIncome_installment"] = safe_ratio(df["annualIncome"], df["installment"] + 0.1)

    bin_sources = [
        ("employmentLength", "employmentLength_bin", 11),
        ("issue_month_index", "issueDate_bin", 120),
        ("earlies_credit_month_index", "earliesCreditLine_bin", 120),
        ("term", "term_bin", 2),
        ("interestRate", "interestRate_bin", 100),
        ("annualIncome", "annualIncome_bin", 10),
        ("loanAmnt", "loanAmnt_bin", 10),
        ("homeOwnership", "homeOwnership_bin", 10),
        ("dti", "dti_bin", 100),
        ("installment", "installment_bin", 100),
        ("revolBal", "revolBal_bin", 100),
        ("revolUtil", "revolUtil_bin", 100),
    ]
    for source_col, feature_name, bins in bin_sources:
        if source_col in {"employmentLength", "issue_month_index", "earlies_credit_month_index", "term", "homeOwnership"}:
            df[feature_name] = df[source_col].fillna(-1).astype("int32")
            forum_cat_cols.append(feature_name)
            continue
        created = add_quantile_bin_feature(df, source_col, feature_name, bins)
        if created is not None:
            forum_cat_cols.append(created)

    for value_col in FORUM_RATIO_FEATURES:
        add_issue_date_window_features(df, value_col)
        add_group_median_ratio_features(
            df,
            value_col,
            "employmentLength",
            f"{value_col}_employmentLength_ratio",
        )
        add_group_median_ratio_features(df, value_col, "purpose", f"{value_col}_purpose_ratio")
        add_group_median_ratio_features(
            df,
            value_col,
            "homeOwnership",
            f"{value_col}_homeOwnership_ratio",
        )

    drop_cols = [col for col in FORUM_PSI_DROP_COLS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    forum_cat_cols = [col for col in dict.fromkeys(forum_cat_cols) if col in df.columns]
    return df, forum_cat_cols


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_category_combos: bool,
    use_numeric_category_cols: bool,
    use_forum_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    y = train["isDefault"].astype("int8").to_numpy()
    train_x = train.drop(columns=["isDefault"])
    all_data = pd.concat([train_x, test], axis=0, ignore_index=True)

    all_data["employmentLength"] = employment_length_to_years(all_data["employmentLength"])
    all_data = add_date_features(all_data)
    all_data = add_grade_features(all_data)
    all_data = add_numeric_features(all_data)
    all_data = add_anonymous_aggregate_features(all_data)

    forum_cat_cols: list[str] = []
    if use_forum_features:
        all_data, forum_cat_cols = add_forum_features(all_data)

    combo_cols: list[str] = []
    if use_category_combos:
        all_data, combo_cols = add_category_combo_features(all_data)

    count_cols = [
        "employmentTitle",
        "postCode",
        "title",
        "regionCode",
        "purpose",
        "grade",
        "subGrade",
        "homeOwnership",
        "verificationStatus",
    ] + combo_cols + forum_cat_cols
    all_data = add_count_features(all_data, count_cols)
    all_data = add_group_stat_features(all_data)

    drop_cols = ["id", "issueDate", "earliesCreditLine", "policyCode"]
    all_data = all_data.drop(columns=[col for col in drop_cols if col in all_data.columns])

    cat_cols = all_data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if use_numeric_category_cols:
        cat_cols = sorted(
            set(cat_cols + combo_cols + forum_cat_cols + [col for col in BASE_CAT_COLS if col in all_data.columns])
        )
    for col in cat_cols:
        all_data[col] = all_data[col].astype("string").fillna("__MISSING__").astype(str)

    for col in all_data.columns:
        if col not in cat_cols and pd.api.types.is_float_dtype(all_data[col]):
            all_data[col] = all_data[col].astype("float32")

    X = all_data.iloc[: len(train)].reset_index(drop=True)
    X_test = all_data.iloc[len(train) :].reset_index(drop=True)
    return X, X_test, y, cat_cols


def main() -> None:
    args = parse_args()
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
    X, X_test, y, cat_cols = build_features(
        train,
        test,
        args.category_combos,
        args.numeric_category_cols,
        args.forum_features,
    )
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    print(f"Feature shape: {X.shape}, Test feature shape: {X_test.shape}, cat cols: {len(cat_cols)}")

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
        "thread_count": args.thread_count,
        "allow_writing_files": False,
        "verbose": 100,
    }

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_scores: list[float] = []
    feature_importances = pd.DataFrame({"feature": X.columns})

    test_pool = Pool(X_test, cat_features=cat_indices)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{args.n_splits}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        valid_pred = model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(float(fold_auc))
        oof[valid_idx] = valid_pred.astype(np.float32)
        test_pred += model.predict_proba(test_pool)[:, 1].astype(np.float32) / args.n_splits
        feature_importances[f"fold_{fold}"] = model.get_feature_importance()
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    elapsed = time.time() - start

    submission_path = output_dir / f"submission_{args.run_name}.csv"
    oof_path = output_dir / f"oof_{args.run_name}.csv"
    importance_path = output_dir / f"feature_importance_{args.run_name}.csv"
    metrics_path = output_dir / f"metrics_{args.run_name}.json"

    pd.DataFrame({"id": test_ids, "isDefault": test_pred}).to_csv(submission_path, index=False)
    pd.DataFrame({"isDefault": y, "pred": oof}).to_csv(oof_path, index=False)
    feature_importances["importance_mean"] = feature_importances.filter(like="fold_").mean(axis=1)
    feature_importances.sort_values("importance_mean", ascending=False).to_csv(importance_path, index=False)

    metrics = {
        "run_name": args.run_name,
        "cv_auc": float(cv_auc),
        "fold_auc": fold_scores,
        "elapsed_seconds": round(elapsed, 2),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": int(X.shape[1]),
        "cat_cols": cat_cols,
        "submission_path": str(submission_path),
        "oof_path": str(oof_path),
        "importance_path": str(importance_path),
        "params": params,
        "numeric_category_cols": args.numeric_category_cols,
        "category_combos": args.category_combos,
        "forum_features": args.forum_features,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"CV AUC: {cv_auc:.6f}")
    print(f"Fold AUC: {[round(v, 6) for v in fold_scores]}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Submission: {submission_path}")
    print(f"OOF: {oof_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
