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
    return parser.parse_args()


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_category_combos: bool,
    use_numeric_category_cols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    y = train["isDefault"].astype("int8").to_numpy()
    train_x = train.drop(columns=["isDefault"])
    all_data = pd.concat([train_x, test], axis=0, ignore_index=True)

    all_data["employmentLength"] = employment_length_to_years(all_data["employmentLength"])
    all_data = add_date_features(all_data)
    all_data = add_grade_features(all_data)
    all_data = add_numeric_features(all_data)
    all_data = add_anonymous_aggregate_features(all_data)

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
    ] + combo_cols
    all_data = add_count_features(all_data, count_cols)
    all_data = add_group_stat_features(all_data)

    drop_cols = ["id", "issueDate", "earliesCreditLine", "policyCode"]
    all_data = all_data.drop(columns=[col for col in drop_cols if col in all_data.columns])

    cat_cols = all_data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if use_numeric_category_cols:
        cat_cols = sorted(set(cat_cols + combo_cols + [col for col in BASE_CAT_COLS if col in all_data.columns]))
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
    X, X_test, y, cat_cols = build_features(train, test, args.category_combos, args.numeric_category_cols)
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
