from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend OOF and submission predictions.")
    parser.add_argument("--oof", nargs="+", required=True, help="OOF csv files with columns isDefault,pred")
    parser.add_argument("--sub", nargs="+", required=True, help="Submission csv files with columns id,isDefault")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional blend weights")
    parser.add_argument("--search-step", type=float, default=0.0, help="Grid-search weights with this step if > 0")
    parser.add_argument("--output-dir", default="outputs_blend", help="Output directory")
    parser.add_argument("--run-name", default="blend", help="Output file prefix")
    return parser.parse_args()


def normalize_weights(weights: list[float] | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    weights_array = np.asarray(weights, dtype=np.float64)
    if np.any(weights_array < 0):
        raise ValueError("Weights must be non-negative")
    total = weights_array.sum()
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return weights_array / total


def blend_oof(paths: list[str], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true, preds = load_oof_matrix(paths)
    return y_true, preds @ weights


def load_oof_matrix(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y_true: np.ndarray | None = None
    preds: list[np.ndarray] = []

    for path in paths:
        df = pd.read_csv(path)
        if not {"isDefault", "pred"}.issubset(df.columns):
            raise ValueError(f"{path} must contain isDefault and pred columns")
        if y_true is None:
            y_true = df["isDefault"].to_numpy()
        elif not np.array_equal(y_true, df["isDefault"].to_numpy()):
            raise ValueError(f"{path} has a different target order")
        preds.append(df["pred"].to_numpy(dtype=np.float64))

    if y_true is None or not preds:
        raise ValueError("No OOF files provided")
    return y_true, np.column_stack(preds)


def iter_weight_grid(n: int, step: float) -> list[np.ndarray]:
    if n < 2:
        return [np.ones(1, dtype=np.float64)]
    if step <= 0 or step > 1:
        raise ValueError("--search-step must be in (0, 1]")

    units = int(round(1 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--search-step must divide 1.0 evenly, e.g. 0.1, 0.05, 0.02")

    results: list[np.ndarray] = []

    def backtrack(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            results.append(np.asarray(prefix + [remaining], dtype=np.float64) / units)
            return
        for value in range(remaining + 1):
            backtrack(prefix + [value], remaining - value, slots - 1)

    backtrack([], units, n)
    return results


def search_best_weights(paths: list[str], step: float) -> tuple[np.ndarray, float]:
    y_true, preds = load_oof_matrix(paths)
    best_auc = -np.inf
    best_weights: np.ndarray | None = None

    for weights in iter_weight_grid(preds.shape[1], step):
        auc = roc_auc_score(y_true, preds @ weights)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("No weights were searched")
    return best_weights, float(best_auc)


def blend_submission(paths: list[str], weights: np.ndarray) -> pd.DataFrame:
    ids: pd.Series | None = None
    blended: np.ndarray | None = None

    for path, weight in zip(paths, weights):
        df = pd.read_csv(path)
        if not {"id", "isDefault"}.issubset(df.columns):
            raise ValueError(f"{path} must contain id and isDefault columns")
        if ids is None:
            ids = df["id"].copy()
            blended = np.zeros(len(df), dtype=np.float64)
        elif not ids.equals(df["id"]):
            raise ValueError(f"{path} has a different id order")
        blended += weight * df["isDefault"].to_numpy(dtype=np.float64)

    if ids is None or blended is None:
        raise ValueError("No submission files provided")
    return pd.DataFrame({"id": ids, "isDefault": blended.astype(np.float32)})


def main() -> None:
    args = parse_args()
    if len(args.oof) != len(args.sub):
        raise ValueError("The number of OOF files must match the number of submission files")

    searched_auc: float | None = None
    if args.search_step > 0:
        weights, searched_auc = search_best_weights(args.oof, args.search_step)
    else:
        weights = normalize_weights(args.weights, len(args.oof))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, oof_pred = blend_oof(args.oof, weights)
    auc = roc_auc_score(y_true, oof_pred)
    submission = blend_submission(args.sub, weights)

    oof_path = output_dir / f"oof_{args.run_name}.csv"
    sub_path = output_dir / f"submission_{args.run_name}.csv"
    metrics_path = output_dir / f"metrics_{args.run_name}.json"

    pd.DataFrame({"isDefault": y_true, "pred": oof_pred.astype(np.float32)}).to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    metrics = {
        "run_name": args.run_name,
        "auc": float(auc),
        "oof": args.oof,
        "sub": args.sub,
        "weights": weights.tolist(),
        "search_step": args.search_step if args.search_step > 0 else None,
        "searched_auc": searched_auc,
        "oof_path": str(oof_path),
        "submission_path": str(sub_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Blend AUC: {auc:.6f}")
    print(f"Weights: {[round(w, 6) for w in weights]}")
    print(f"OOF: {oof_path}")
    print(f"Submission: {sub_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
