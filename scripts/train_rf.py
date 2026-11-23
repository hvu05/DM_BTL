#!/usr/bin/env python3
"""Train, tune (5-fold CV), evaluate and explain a Random Forest on SOHAS HOG features.

Random Forest works directly on the 26,244 raw HOG dimensions (trees need no scaling, and
keeping the raw features lets feature importances be mapped back to image regions).

Hyper-parameters tuned with stratified 5-fold CV on the training rows (Macro-F1):
  * n_estimators : number of trees            (RF_N_ESTIMATORS)
  * max_depth    : maximum depth of each tree (RF_MAX_DEPTH, None = grow until pure)
  * class_weight : None / "balanced" / "balanced_subsample" (counters the 8.9:1 imbalance)
For each (max_depth, class_weight) pair a single forest with max(n_estimators) trees is fitted
per fold; smaller forests are scored from its first k trees. This is exactly the forest that
n_estimators=k would grow with the same random_state, so the grid is searched without refitting.

Outputs (same format as outputs/svm/): metrics summary, per-class report, confusion matrices,
tuning table + curve, misclassified test images, per-class F1 chart, HOG feature-importance map,
and the fitted model (.joblib).

Local usage (after HOG features exist in dataset/features/):
    python scripts/train_rf.py --branches direct padding

The section between the RF CORE markers is copied verbatim into the Kaggle kernel by
scripts/build_rf_kernel.py, so keep it self-contained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --- BEGIN RF CORE ---
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTH = [20, None]
RF_CLASS_WEIGHT = [None, "balanced", "balanced_subsample"]
RF_CV_FOLDS = 5
RF_SEED = 42
# HOG layout produced by extract_hog_branch: 224x224, 8x8 cells, 2x2 cells/block, 9 orientations
RF_HOG_BLOCKS = 27
RF_HOG_CELLS_PER_BLOCK = 2
RF_HOG_ORIENTATIONS = 9


def _rf_model(n_estimators: int, max_depth: int | None, class_weight: str | None, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def _rf_staged_predictions(forest: RandomForestClassifier, X: np.ndarray, checkpoints: list[int]) -> dict[int, np.ndarray]:
    """Predicted labels of the sub-forest made of the first k trees, for every k in checkpoints."""
    proba_sum = np.zeros((X.shape[0], forest.n_classes_), dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    wanted = set(checkpoints)
    for k, tree in enumerate(forest.estimators_, start=1):
        proba_sum += tree.predict_proba(X)
        if k in wanted:
            out[k] = forest.classes_[np.argmax(proba_sum, axis=1)]
    return out


def _cw_label(class_weight: str | None) -> str:
    return "none" if class_weight is None else class_weight


def _depth_label(max_depth: int | None) -> str:
    return "None" if max_depth is None else str(max_depth)


def _rf_metric_row(model: str, branch: str, split_name: str, params: str, y_true: np.ndarray,
                   y_pred: np.ndarray, train_time: float, predict_time: float) -> dict:
    return {
        "model": model,
        "branch": branch,
        "split": split_name,
        "best_hyperparams": params,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "train_time_sec": round(train_time, 2),
        "predict_time_sec": round(predict_time, 2),
    }


def _rf_per_class_rows(model: str, branch: str, split_name: str, y_true: np.ndarray,
                       y_pred: np.ndarray, class_names: list[str]) -> list[dict]:
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    return [{
        "model": model, "branch": branch, "split": split_name, "class_name": c,
        "precision": round(rep[c]["precision"], 4),
        "recall": round(rep[c]["recall"], 4),
        "f1_score": round(rep[c]["f1-score"], 4),
        "support": int(rep[c]["support"]),
    } for c in class_names]


def _rf_plot_confusion(cm: np.ndarray, classes: list[str], title: str, output_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_display = np.divide(cm.astype(float), row_sums, out=np.zeros(cm.shape), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    cax = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, title=title,
           ylabel="True Label", xlabel="Predicted Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_display[i, j]:.1%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cm_display[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")


def _rf_plot_tuning(tuning_df: pd.DataFrame, branch: str, best: dict, output_path: Path) -> None:
    depths = list(dict.fromkeys(tuning_df["max_depth"]))
    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 4.5), dpi=150, sharey=True, squeeze=False)
    colors = {"none": "#1f77b4", "balanced": "#d62728", "balanced_subsample": "#2ca02c"}
    for ax, depth in zip(axes[0], depths):
        sub = tuning_df[tuning_df["max_depth"] == depth]
        for cw, color in colors.items():
            s = sub[sub["class_weight"] == cw]
            if s.empty:
                continue
            ax.errorbar(s["n_estimators"], s["cv_macro_f1_mean"], yerr=s["cv_macro_f1_std"], marker="o",
                        lw=2, capsize=4, color=color, label=f"CV, class_weight={cw}")
            ax.plot(s["n_estimators"], s["val_macro_f1"], marker="s", lw=1.5, linestyle="--",
                    color=color, alpha=0.7, label=f"Val, class_weight={cw}")
        if depth == best["max_depth"]:
            ax.axvline(best["n_estimators"], color="gray", linestyle=":")
        ax.set_title(f"max_depth = {depth}")
        ax.set_xlabel("n_estimators (number of trees)")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0][0].set_ylabel("Macro-F1")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"Random Forest tuning (branch: {branch}) - chosen: n_estimators={best['n_estimators']}, "
                 f"max_depth={best['max_depth']}, class_weight={best['class_weight']}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved tuning curve: {output_path}")


def _rf_plot_per_class(per_class_df: pd.DataFrame, branch: str, output_path: Path) -> None:
    """Per-class test F1 vs. number of training images: does RF help the small classes?"""
    df = per_class_df.sort_values("train_images")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(df["class_name"], df["f1_score"], color="#1f77b4")
    for bar, n in zip(bars, df["train_images"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n_train={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test F1-score")
    ax.set_title(f"Random Forest per-class F1 on test, classes sorted by training size ({branch})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved per-class chart: {output_path}")


def _rf_plot_importance(importances: np.ndarray, branch: str, output_path: Path) -> np.ndarray:
    """Sum the importance of each HOG block (27x27 grid over the 224x224 image) and draw it."""
    b, c, o = RF_HOG_BLOCKS, RF_HOG_CELLS_PER_BLOCK, RF_HOG_ORIENTATIONS
    if importances.size != b * b * c * c * o:
        print(f"[skip] importance map: unexpected HOG size {importances.size}")
        return np.empty((0, 0))
    block_map = importances.reshape(b, b, c, c, o).sum(axis=(2, 3, 4))
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=150)
    im = ax.imshow(block_map, cmap="hot", interpolation="nearest", extent=(0, 224, 224, 0))
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Summed Gini importance")
    ax.set_title(f"Random Forest HOG importance by image region ({branch})")
    ax.set_xlabel("x (pixels in the 224x224 image)")
    ax.set_ylabel("y (pixels)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved importance map: {output_path}")
    return block_map


def train_and_eval_rf(
    features_dir: Path,
    output_dir: Path,
    branches: list[str] | tuple[str, ...] = ("direct", "padding"),
    cv_folds: int = RF_CV_FOLDS,
    seed: int = RF_SEED,
) -> pd.DataFrame:
    """Tune n_estimators/max_depth/class_weight with stratified K-fold CV on train; report val + test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "RandomForest"
    checkpoints = sorted(RF_N_ESTIMATORS)
    n_max = checkpoints[-1]
    combos = [(d, cw) for d in RF_MAX_DEPTH for cw in RF_CLASS_WEIGHT]
    summaries: list[dict] = []
    per_class: list[dict] = []

    for branch in branches:
        npz_file = features_dir / f"hog_{branch}.npz"
        if not npz_file.exists():
            print(f"[skip] Feature file not found: {npz_file}")
            continue
        print("=" * 50)
        print(f" RANDOM FOREST ON BRANCH: {branch.upper()}")
        print("=" * 50)

        with np.load(npz_file, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            split = data["split"].astype(str)
            class_names = [str(c) for c in data["class_names"]]
            image_ids = np.asarray([str(v) for v in data["image_ids"]])
            relative_paths = np.asarray([str(v) for v in data["relative_paths"]])

        tr_m, va_m, te_m = split == "train", split == "val", split == "test"
        X_train, y_train = np.ascontiguousarray(X[tr_m], dtype=np.float32), y[tr_m]
        X_val, y_val = np.ascontiguousarray(X[va_m], dtype=np.float32), y[va_m]
        X_test, y_test = np.ascontiguousarray(X[te_m], dtype=np.float32), y[te_m]
        del X
        print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)} | HOG dim={X_train.shape[1]}")

        # ---- 1. Stratified K-fold CV on train ----
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores: dict[tuple, list[float]] = {}
        total_fits, done, t_cv = cv_folds * len(combos), 0, time.time()
        for fold, (fit_idx, hold_idx) in enumerate(skf.split(X_train, y_train), start=1):
            for depth, cw in combos:
                forest = _rf_model(n_max, depth, cw, seed).fit(X_train[fit_idx], y_train[fit_idx])
                staged = _rf_staged_predictions(forest, X_train[hold_idx], checkpoints)
                for k, pred in staged.items():
                    scores.setdefault((k, depth, cw), []).append(
                        f1_score(y_train[hold_idx], pred, average="macro", zero_division=0))
                done += 1
                elapsed = time.time() - t_cv
                print(f"  fold {fold}/{cv_folds} max_depth={_depth_label(depth)} class_weight={_cw_label(cw)} "
                      f"-> Macro-F1@{n_max}={scores[(n_max, depth, cw)][-1]:.4f} "
                      f"[{done}/{total_fits}, ETA {elapsed / done * (total_fits - done) / 60:.1f} min]", flush=True)
                del forest

        # ---- 2. Same grid fitted on full train, scored on val (for reference / overfitting check) ----
        tuning_rows = []
        for depth, cw in combos:
            forest = _rf_model(n_max, depth, cw, seed).fit(X_train, y_train)
            staged = _rf_staged_predictions(forest, X_val, checkpoints)
            for k in checkpoints:
                cv = scores[(k, depth, cw)]
                tuning_rows.append({
                    "branch": branch, "n_estimators": k, "max_depth": _depth_label(depth),
                    "class_weight": _cw_label(cw),
                    "cv_macro_f1_mean": round(float(np.mean(cv)), 4),
                    "cv_macro_f1_std": round(float(np.std(cv)), 4),
                    "val_macro_f1": round(f1_score(y_val, staged[k], average="macro", zero_division=0), 4),
                    "val_accuracy": round(accuracy_score(y_val, staged[k]), 4),
                })
            del forest
        tuning_df = pd.DataFrame(tuning_rows)
        tuning_df.to_csv(output_dir / f"tuning_rf_{branch}.csv", index=False)
        print(tuning_df.to_string(index=False))

        # Highest mean CV Macro-F1; ties -> fewer trees (cheaper model).
        best = tuning_df.sort_values(["cv_macro_f1_mean", "n_estimators"], ascending=[False, True]).iloc[0].to_dict()
        best_depth = None if best["max_depth"] == "None" else int(best["max_depth"])
        best_cw = None if best["class_weight"] == "none" else best["class_weight"]
        best_n = int(best["n_estimators"])
        print(f"Chosen n_estimators={best_n}, max_depth={best['max_depth']}, class_weight={best['class_weight']} "
              f"(CV Macro-F1={best['cv_macro_f1_mean']:.4f}±{best['cv_macro_f1_std']:.4f})")
        _rf_plot_tuning(tuning_df, branch, best, output_dir / f"val_curve_rf_{branch}.png")

        # ---- 3. Final model on train with the chosen setting ----
        t0 = time.time()
        final_rf = _rf_model(best_n, best_depth, best_cw, seed).fit(X_train, y_train)
        train_time = time.time() - t0
        joblib.dump(final_rf, output_dir / f"rf_{branch}_model.joblib", compress=3)

        params = f"n_estimators={best_n},max_depth={best['max_depth']},class_weight={best['class_weight']}"
        for split_name, X_s, y_s in (("val", X_val, y_val), ("test", X_test, y_test)):
            t0 = time.time()
            pred = final_rf.predict(X_s)
            predict_time = time.time() - t0
            summaries.append(_rf_metric_row(model_name, branch, split_name, params, y_s, pred, train_time, predict_time))
            per_class.extend(_rf_per_class_rows(model_name, branch, split_name, y_s, pred, class_names))

        test_proba = final_rf.predict_proba(X_test)
        test_pred = final_rf.classes_[np.argmax(test_proba, axis=1)]
        _rf_plot_confusion(
            confusion_matrix(y_test, test_pred, labels=list(range(len(class_names)))), class_names,
            f"Random Forest - Test ({branch.title()})", output_dir / f"confusion_matrix_rf_{branch}.png",
        )

        test_pc = pd.DataFrame(_rf_per_class_rows(model_name, branch, "test", y_test, test_pred, class_names))
        test_pc["train_images"] = [int(np.sum(y_train == i)) for i in range(len(class_names))]
        _rf_plot_per_class(test_pc, branch, output_dir / f"per_class_f1_rf_{branch}.png")

        wrong = test_pred != y_test
        pd.DataFrame({
            "image_id": image_ids[te_m][wrong],
            "relative_path": relative_paths[te_m][wrong],
            "true_class": [class_names[i] for i in y_test[wrong]],
            "predicted_class": [class_names[i] for i in test_pred[wrong]],
            "predicted_prob": np.round(test_proba[wrong].max(axis=1), 4),
        }).sort_values("predicted_prob", ascending=False).to_csv(
            output_dir / f"misclassifications_rf_{branch}.csv", index=False)
        print(f"Test misclassifications: {int(wrong.sum())}/{len(y_test)}")

        importances = final_rf.feature_importances_
        block_map = _rf_plot_importance(importances, branch, output_dir / f"feature_importance_map_rf_{branch}.png")
        if block_map.size:
            b, c, o = RF_HOG_BLOCKS, RF_HOG_CELLS_PER_BLOCK, RF_HOG_ORIENTATIONS
            top = np.argsort(importances)[::-1][:50]
            br, bc, cr, cc, ori = np.unravel_index(top, (b, b, c, c, o))
            pd.DataFrame({
                "feature_index": top, "importance": np.round(importances[top], 6),
                "block_row": br, "block_col": bc, "cell_row_in_block": cr, "cell_col_in_block": cc,
                "orientation_bin": ori,
                "pixel_y": (br + cr) * 8, "pixel_x": (bc + cc) * 8,
            }).to_csv(output_dir / f"top_features_rf_{branch}.csv", index=False)
            np.save(output_dir / f"importance_block_map_rf_{branch}.npy", block_map)
        del X_train, X_val, X_test, final_rf

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "rf_metrics_summary.csv", index=False)
        pd.DataFrame(per_class).to_csv(output_dir / "rf_per_class_report.csv", index=False)
        print("=" * 50)
        print(" RANDOM FOREST RESULTS")
        print("=" * 50)
        print(summary_df.to_string(index=False))
    return summary_df
# --- END RF CORE ---


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=Path("dataset/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rf"))
    parser.add_argument("--branches", nargs="+", default=["direct", "padding"], choices=["direct", "padding"])
    parser.add_argument("--cv-folds", type=int, default=RF_CV_FOLDS)
    parser.add_argument("--random-state", type=int, default=RF_SEED)
    args = parser.parse_args()
    result = train_and_eval_rf(args.features_dir, args.output_dir, args.branches,
                               cv_folds=args.cv_folds, seed=args.random_state)
    return 0 if not result.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
