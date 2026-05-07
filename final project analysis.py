#!/usr/bin/env python3
"""
CS 135 Final Project — Wisconsin Breast Cancer Diagnostic (WDBC) classification.

Sections map to assignment requirements (1)–(7): load/report, preprocessing/split,
three GridSearchCV-tuned models, printed metrics per model, figures, directory,
final summary table.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe figure rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
FIG_DIR = "figures"

# Canonical label codes for sklearn≥1.x WDBC convention (validated at runtime)
_MALIGNANT_CODE: int = 0
_BENIGN_CODE: int = 1


def detect_class_codes(bundle) -> tuple[int, int]:
    """Map readable pathologist labels ↔ integer encodings supplied by sklearn."""
    names = np.asarray(bundle.target_names, dtype=str)
    mal = int(np.flatnonzero(names == "malignant")[0])
    ben = int(np.flatnonzero(names == "benign")[0])
    uniq = set(int(x) for x in np.unique(bundle.target))
    if uniq != {mal, ben}:
        raise RuntimeError("Unexpected labeling of the sklearn WDBC archive.")
    return mal, ben


def _clinical_metrics(y_true, y_pred):
    malignant_label = _MALIGNANT_CODE
    return (
        accuracy_score(y_true, y_pred),
        precision_score(
            y_true, y_pred, average="binary", pos_label=malignant_label, zero_division=0
        ),
        recall_score(
            y_true, y_pred, average="binary", pos_label=malignant_label, zero_division=0
        ),
        f1_score(
            y_true, y_pred, average="binary", pos_label=malignant_label, zero_division=0
        ),
    )


def _print_metrics(title: str, y_true, y_pred):
    acc, prec, rec, f1 = _clinical_metrics(y_true, y_pred)
    print(f"  {title} accuracy : {acc:.4f}")
    print(f"  {title} precision: {prec:.4f} (malignant positive)")
    print(f"  {title} recall   : {rec:.4f}")
    print(f"  {title} F1       : {f1:.4f}")
    return acc, prec, rec, f1


def setup_style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    colors = sns.color_palette("colorblind", as_cmap=False)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def fig1_hyperparam_validation_curves(X_train, y_train, X_val, y_val):
    """
    Requirement (5) — Fig.1 validation accuracy surfaces on the held-out 15% slice.

    Every configuration is trained only on train, scored on validation.
    """
    log_c = []
    lr_val_acc = []
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        log_c.append(np.log10(c))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(X_train, y_train)
        lr_val_acc.append(accuracy_score(y_val, pipe.predict(X_val)))

    rf_depths_internal: list[int | None] = [None, 5, 10]
    rf_depth_labels = ["None", "5", "10"]
    rf_n_est = [50, 100, 200]
    rf_acc = np.zeros((len(rf_n_est), len(rf_depths_internal)))
    for i, ne in enumerate(rf_n_est):
        for j, md in enumerate(rf_depths_internal):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=ne,
                            max_depth=md,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            rf_acc[i, j] = accuracy_score(y_val, pipe.predict(X_val))

    svm_c_vals = [0.1, 1.0, 10.0]
    svm_gamma_vals: list[float | str] = ["scale", "auto", 0.01]
    svm_gamma_ticks = [str(g) for g in svm_gamma_vals]
    svm_acc = np.zeros((len(svm_c_vals), len(svm_gamma_vals)))
    for i, cc in enumerate(svm_c_vals):
        for j, gg in enumerate(svm_gamma_vals):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=cc, gamma=gg, random_state=RANDOM_STATE)),
                ]
            )
            pipe.fit(X_train, y_train)
            svm_acc[i, j] = accuracy_score(y_val, pipe.predict(X_val))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    ax_lr = axes[0]
    ax_lr.plot(log_c, lr_val_acc, marker="o", lw=2, color=sns.color_palette("colorblind")[0])
    ax_lr.set_xticks(log_c)
    ax_lr.set_xticklabels(["0.01", "0.1", "1", "10", "100"], rotation=0)
    ax_lr.set_xlabel("log-scaled ridge strength (log10(C) for logistic regression)")
    ax_lr.set_ylabel("Validation accuracy")
    ax_lr.set_title("Logistic Regression — val accuracy vs C")

    sns.heatmap(
        rf_acc,
        ax=axes[1],
        cmap="Blues",
        cbar_kws={"label": "Val acc"},
        xticklabels=rf_depth_labels,
        yticklabels=rf_n_est,
        annot=True,
        fmt=".3f",
    )
    axes[1].set_title("Random Forest")
    axes[1].set_xlabel("max_depth (None shown as literal string)")
    axes[1].set_ylabel("n_estimators")

    sns.heatmap(
        svm_acc,
        ax=axes[2],
        cmap="Blues",
        cbar_kws={"label": "Val acc"},
        xticklabels=svm_gamma_ticks,
        yticklabels=[str(c) for c in svm_c_vals],
        annot=True,
        fmt=".3f",
    )
    axes[2].set_title("SVM (RBF)")
    axes[2].set_xlabel("gamma")
    axes[2].set_ylabel("C")

    fig.savefig(os.path.join(FIG_DIR, "fig1_hyperparam.png"), dpi=200)
    plt.close(fig)


def fig2_grouped_comparison(model_rows: pd.DataFrame):
    """Grouped bar chart of test classification metrics."""
    fig, ax = plt.subplots(figsize=(9.2, 5.1), constrained_layout=True)
    sns.barplot(
        data=model_rows,
        x="metric",
        y="score",
        hue="model",
        dodge=True,
        palette="colorblind",
        ax=ax,
    )
    ax.set_title("Model comparison — test set (positive class = malignant)")
    ax.set_ylim(0.0, 1.08)
    fig.savefig(os.path.join(FIG_DIR, "fig2_comparison.png"), dpi=200)
    plt.close(fig)


def fig3_confusion_matrices(models: dict[str, Pipeline], X_test, y_test):
    labels_order = [_MALIGNANT_CODE, _BENIGN_CODE]
    display_labels = ["Malignant", "Benign"]
    titles = [
        name
        for name in ("Logistic Regression", "Random Forest", "SVM (RBF)")
        if name in models
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    for ax, title in zip(axes, titles, strict=True):
        estimator = models[title]
        y_hat = estimator.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_hat,
            labels=labels_order,
            display_labels=display_labels,
            cmap="Blues",
            normalize=None,
            colorbar=False,
            ax=ax,
        )
        ax.set_title(title)

    fig.savefig(os.path.join(FIG_DIR, "fig3_confusion.png"), dpi=200)
    plt.close(fig)


def derive_importances(
    estimator: Pipeline, X_ref, y_ref
) -> tuple[np.ndarray, str]:
    """Return nonnegative importance vector spanning all cytology-derived features."""
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        scores = np.abs(np.asarray(clf.feature_importances_))
        note = "|RF importance|"
    elif hasattr(clf, "coef_"):
        scores = np.abs(np.asarray(clf.coef_.ravel()))
        note = "|logistic coef|"
    else:
        permuted = permutation_importance(
            estimator,
            X_ref,
            y_ref,
            n_repeats=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scores = np.abs(permuted.importances_mean)
        note = "permutation Δ (kernel SVM proxy)"

    if scores.shape[0] != X_ref.shape[1]:
        raise ValueError("Importance dimensionality mismatch.")

    scores = scores.astype(float, copy=False)
    return scores, note


def fig4_importance_plot(
    best_name: str, estimator: Pipeline, X_train, y_train, feat_names
):
    imp_raw, analytic_tag = derive_importances(estimator, X_train, y_train)
    order_all = np.argsort(imp_raw)
    strongest_idx = order_all[-10:][::-1]
    vals = imp_raw[strongest_idx]
    labels_plot = feat_names[strongest_idx]

    fig, ax = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    ax.barh(
        np.arange(len(vals)),
        vals[::-1],
        color=sns.color_palette("colorblind", n_colors=len(vals)),
    )
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labels_plot[::-1], fontsize=9)
    ax.set_title(
        f"{best_name} diagnostics\nTop-N drivers ({analytic_tag})",
        fontsize=12,
        loc="left",
    )
    ax.set_xlabel("Relative influence (scaled per method)")
    fig.savefig(os.path.join(FIG_DIR, "fig4_error_analysis.png"), dpi=200)
    plt.close(fig)
    return strongest_idx[:5]


def textual_error_audit(
    best_name: str,
    estimator: Pipeline,
    X_test,
    y_test,
    top5_idx: np.ndarray,
    feat_names: np.ndarray,
):
    y_hat = estimator.predict(X_test)
    mal, ben = _MALIGNANT_CODE, _BENIGN_CODE

    fp_total = fn_total = 0  # malignant-positive ontology
    for yt, yp in zip(y_test, y_hat, strict=True):
        fn_total += int(yt == mal and yp == ben)
        fp_total += int(yt == ben and yp == mal)

    mis_idx = np.where(y_hat != y_test)[0]
    ok_idx = np.where(y_hat == y_test)[0]

    print("\nDetailed error characterization (positive = malignant specimen)")
    print(f" Selected model :: {best_name}")
    print(f" False negatives (missed malignant): {fn_total}")
    print(f" False positives (benign over-called): {fp_total}")

    top5_feats = feat_names[list(top5_idx)]

    cols = pd.Index(feat_names, dtype=str)

    Frame_ok = pd.DataFrame(X_test[ok_idx], columns=cols)
    Frame_bad = pd.DataFrame(X_test[mis_idx], columns=cols)

    stacked = []
    for fname in top5_feats:
        stacked.append(
            {
                "feature": fname,
                "mean_correct": float(Frame_ok[fname].mean()) if ok_idx.size else np.nan,
                "mean_wrong": float(Frame_bad[fname].mean()) if mis_idx.size else np.nan,
            }
        )

    stacked_df = pd.DataFrame(stacked).set_index("feature")
    stacked_df["difference_wrong_minus_correct"] = (
        stacked_df["mean_wrong"] - stacked_df["mean_correct"]
    )

    pd.set_option("display.width", 130)
    print("\nMisclassified-vs-correct means for top-ranked drivers:")
    print(stacked_df.to_string(float_format=lambda v: f"{v:+.5f}"))
    print(
        "\nCardinality — misclassified test rows:",
        int(mis_idx.size),
        " correct:",
        int(ok_idx.size),
    )


def train_grids(X_train, y_train) -> dict[str, GridSearchCV]:
    """Requirement (3) — GridSearchCV with five folds on raw train tensors."""
    blueprints = {
        "Logistic Regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=6000, random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 5, 10],
            },
        ),
        "SVM (RBF)": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto", 0.01]},
        ),
    }

    fitted: dict[str, GridSearchCV] = {}
    for label, (pipe, grid) in blueprints.items():
        grid_search = GridSearchCV(
            pipe,
            grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid_search.fit(X_train, y_train)
        fitted[label] = grid_search

    return fitted


def pprint_params(params: dict) -> str:
    pieces = []
    for key in sorted(params):
        val = params[key]
        if val is None:
            pretty = "None"
        elif isinstance(val, float):
            pretty = f"{val:g}"
        else:
            pretty = repr(val).replace("'", "")
        pieces.append(f"{key}:{pretty}")
    return " | ".join(pieces)


def main():
    global _MALIGNANT_CODE, _BENIGN_CODE
    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------ (1)
    bundle = load_breast_cancer()
    feat_names = np.asarray(bundle.feature_names, dtype=str)
    X = np.asarray(bundle.data, dtype=float)
    y = np.asarray(bundle.target, dtype=int)
    _MALIGNANT_CODE, _BENIGN_CODE = detect_class_codes(bundle)

    print("=== (1) Wisconsin Breast Cancer — class balance ===")
    for code in sorted(np.unique(y)):
        nm = str(bundle.target_names[int(code)])
        count = int(np.sum(y == code))
        pct = 100.0 * count / len(y)
        print(f"{nm:>12s} (label {code}): {count} cases ({pct:.1f}%)")

    # ------------------------------------------------------------------ (2)
    (
        X_train,
        X_remain,
        y_train,
        y_remain,
    ) = train_test_split(
        X,
        y,
        train_size=0.70,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remain,
        y_remain,
        test_size=0.50,
        stratify=y_remain,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print("\n=== (2) Preprocessing & splits ===")
    print("StandardScaler is fit inside each estimator pipeline using train folds only.")
    print(
        "Split sizes — train:",
        X_train.shape[0],
        "| val:",
        X_val.shape[0],
        "| test:",
        X_test.shape[0],
    )

    # ------------------------------------------------------------------ (3)
    print("\n=== (3) Hyperparameter search (5-fold CV on train-only tensors) ===")
    searches = train_grids(X_train, y_train)

    # ------------------------------------------------------------------ (4)
    print("\n=== (4) Held-out validation / test metrics ===")
    pipelines: dict[str, Pipeline] = {}
    val_f1_scores: dict[str, float] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    ordered_names = ("Logistic Regression", "Random Forest", "SVM (RBF)")

    for name in ordered_names:
        estimator: Pipeline = searches[name].best_estimator_
        print(f"\n--- {name} ---")
        print("Best hyper-parameters:", searches[name].best_params_)

        _, _, _, val_f1 = _print_metrics(
            "Validation", y_val, estimator.predict(X_val)
        )
        acc, prec, rec, f1 = _print_metrics(
            "Test", y_test, estimator.predict(X_test)
        )

        pipelines[name] = estimator
        val_f1_scores[name] = val_f1
        test_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    fig1_hyperparam_validation_curves(X_train, y_train, X_val, y_val)

    long_rows = []
    for name in ordered_names:
        for metric_key, metric_val in test_metrics[name].items():
            metric_label = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
            }[metric_key]
            long_rows.append(
                {"model": name, "metric": metric_label, "score": metric_val}
            )
    fig2_grouped_comparison(pd.DataFrame(long_rows))
    fig3_confusion_matrices(pipelines, X_test, y_test)

    # ------------------------------------------------------------------ Champion for Figure 4
    leaderboard = sorted(
        ordered_names,
        key=lambda nm: (
            -test_metrics[nm]["accuracy"],
            -test_metrics[nm]["f1"],
            -val_f1_scores[nm],
        ),
    )
    champ_name = leaderboard[0]
    champ_estimator = pipelines[champ_name]
    print(
        "\nFigure 4 deep-dive model (tie-order: highest test accuracy, then test "
        "F1, then validation F1):",
        champ_name,
    )

    top5_idx = fig4_importance_plot(
        champ_name, champ_estimator, X_train, y_train, feat_names
    )
    textual_error_audit(
        champ_name, champ_estimator, X_test, y_test, top5_idx, feat_names
    )

    # ------------------------------------------------------------------ Summary table requirements (6)-(7)
    print("\n=== Final summary ===")
    summary_rows = []
    for name in ordered_names:
        summary_rows.append(
            {
                "model": name,
                "best_params": pprint_params(searches[name].best_params_),
                "val F1": val_f1_scores[name],
                "test F1": test_metrics[name]["f1"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig_paths = sorted(
        os.path.join(FIG_DIR, fname)
        for fname in os.listdir(FIG_DIR)
        if fname.lower().endswith(".png")
    )
    print("\nSaved visuals:", "; ".join(fig_paths))


if __name__ == "__main__":
    main()
