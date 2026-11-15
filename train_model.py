"""
train_model.py
--------------
Trains three classifiers (Logistic Regression, Decision Tree, Random Forest)
on a phishing-URL dataset, evaluates them, selects the best model by F1-score,
and saves it to  models/model.pkl  together with metadata.

Dataset strategy (in priority order):
  1. UCI Phishing Websites dataset  — place the downloaded .arff or .csv file
     in the  dataset/  folder and name it  phishing.arff  or  phishing.csv.
  2. Synthetic dataset built from real URL patterns (fallback, no download
     needed) — clearly labelled so results cannot be misrepresented.

Run:
    python train_model.py

Outputs:
    models/model.pkl          – best model + metadata (joblib)
    reports/evaluation.txt    – text report with all metrics
    reports/confusion_matrix.png
    reports/roc_curve.png
    reports/feature_importance.png
"""

import os
import sys
import time
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# Local module
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import FEATURE_NAMES, extract_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "model.pkl")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

def _load_uci_arff(path: str) -> pd.DataFrame:
    """Load UCI phishing dataset from .arff file."""
    from scipy.io import arff
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode byte strings produced by scipy arff reader
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    # UCI target column is 'Result': '1' = legitimate, '-1' = phishing
    df["label"] = df["Result"].apply(lambda x: 0 if str(x).strip() == "-1" else 1)
    return df


def _load_uci_csv(path: str) -> pd.DataFrame:
    """Load a pre-converted UCI phishing CSV (last column = label)."""
    df = pd.read_csv(path)
    last_col = df.columns[-1]
    df["label"] = df[last_col].apply(lambda x: 0 if int(x) == -1 else 1)
    return df


def _build_synthetic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Build a reproducible synthetic dataset with realistic class overlap,
    based on URL-pattern statistics from published phishing research
    (Mohammad et al. 2015, Sahingoz et al. 2019).

    Feature distributions have intentional overlap so models achieve
    realistic (not perfect) accuracy on real-world URLs.

    Label: 1 = legitimate, 0 = phishing
    """
    rng = np.random.default_rng(RANDOM_STATE)

    def _sample(n, low, high, noise_std=0):
        """Sample integers with optional Gaussian noise for overlap."""
        base = rng.integers(low, high + 1, n).astype(float)
        if noise_std > 0:
            base += rng.normal(0, noise_std, n)
        return np.clip(base, 0, high * 2).astype(int)

    def _bern(n, p):
        """Bernoulli samples."""
        return (rng.random(n) < p).astype(int)

    def _phishing_rows(n):
        # Phishing URL statistics based on Mohammad et al. (2015) and
        # Sahingoz et al. (2019) phishing URL datasets.
        # Realistic url_length range: 20-120 (median ~54 for phishing)
        # Binary features are the strongest discriminators.
        return {
            "url_length":           _sample(n, 20, 120, noise_std=20),
            "hostname_length":      _sample(n, 10, 50,  noise_std=8),
            "path_length":          _sample(n, 5,  70,  noise_std=12),
            "num_dots":             _sample(n, 1,  6,   noise_std=1),
            "num_hyphens":          _sample(n, 1,  5,   noise_std=1),
            "num_at":               _sample(n, 0,  2,   noise_std=0),
            "num_question_marks":   _sample(n, 0,  3,   noise_std=0),
            "num_equal":            _sample(n, 0,  4,   noise_std=0),
            "num_slashes":          _sample(n, 2,  9,   noise_std=2),
            "num_percent":          _sample(n, 0,  3,   noise_std=1),
            # KEY binary discriminators — strongly skewed for phishing
            "has_https":            _bern(n, 0.30),  # only 30% phishing use HTTPS
            "has_ip":               _bern(n, 0.20),  # 20% use raw IP
            "has_suspicious_tld":   _bern(n, 0.45),  # 45% use suspicious TLD
            "num_subdomains":       _sample(n, 1, 5, noise_std=1),
            "has_phishing_keyword": _bern(n, 0.70),  # 70% contain phishing keywords
            "double_slash_in_path": _bern(n, 0.12),
            "label":                np.zeros(n, dtype=int),
        }

    def _legit_rows(n):
        # Legitimate URL statistics (median url_length ~23)
        return {
            "url_length":           _sample(n, 10, 80,  noise_std=15),
            "hostname_length":      _sample(n, 4,  30,  noise_std=6),
            "path_length":          _sample(n, 0,  50,  noise_std=10),
            "num_dots":             _sample(n, 1,  4,   noise_std=1),
            "num_hyphens":          _sample(n, 0,  2,   noise_std=1),
            "num_at":               _bern(n, 0.01),    # very rare
            "num_question_marks":   _sample(n, 0,  2,   noise_std=0),
            "num_equal":            _sample(n, 0,  2,   noise_std=0),
            "num_slashes":          _sample(n, 2,  6,   noise_std=1),
            "num_percent":          _bern(n, 0.04),    # rare
            # KEY binary discriminators — strongly skewed for legitimate
            "has_https":            _bern(n, 0.88),   # 88% legit use HTTPS
            "has_ip":               _bern(n, 0.01),   # almost never use IP
            "has_suspicious_tld":   _bern(n, 0.02),   # almost never suspicious TLD
            "num_subdomains":       _sample(n, 0, 2, noise_std=1),
            "has_phishing_keyword": _bern(n, 0.12),   # ~12% legit contain some keyword
            "double_slash_in_path": _bern(n, 0.01),
            "label":                np.ones(n, dtype=int),
        }

    n_each = n_samples // 2
    phish_df = pd.DataFrame(_phishing_rows(n_each))
    legit_df  = pd.DataFrame(_legit_rows(n_each))
    df = pd.concat([phish_df, legit_df], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return df


def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Try to load the UCI dataset; fall back to synthetic.
    Returns (DataFrame with FEATURE_NAMES columns + 'label', source_description).
    """
    arff_path = os.path.join(DATASET_DIR, "phishing.arff")
    csv_path  = os.path.join(DATASET_DIR, "phishing.csv")

    if os.path.exists(arff_path):
        print(f"[INFO] Loading UCI dataset from {arff_path}")
        try:
            df = _load_uci_arff(arff_path)
            # UCI dataset uses different column names; we need our FEATURE_NAMES
            # For UCI we'll use the numeric columns directly (drop 'Result','label')
            feature_cols = [c for c in df.columns if c not in ("Result", "label")]
            # Map UCI columns to our features where possible, else synthesise
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            y = df["label"]
            source = "UCI Phishing Websites Dataset (archive.ics.uci.edu/dataset/327)"
            # Return with generic column names (model trained on UCI feature set)
            df_out = X.copy()
            df_out["label"] = y
            return df_out, source
        except Exception as e:
            print(f"[WARN] Failed to parse ARFF: {e}. Trying CSV...")

    if os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from {csv_path}")
        try:
            df = _load_uci_csv(csv_path)
            feature_cols = [c for c in df.columns if c not in ("label",)]
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            y = df["label"]
            df_out = X.copy()
            df_out["label"] = y
            source = "CSV dataset loaded from dataset/phishing.csv"
            return df_out, source
        except Exception as e:
            print(f"[WARN] Failed to parse CSV: {e}. Using synthetic dataset.")

    print("[INFO] No external dataset found. Building synthetic dataset.")
    print("       (Download UCI dataset to dataset/phishing.arff for real data)")
    df = _build_synthetic_dataset()
    source = (
        "Synthetic dataset (URL pattern statistics derived from published "
        "phishing research). Real accuracy may differ on live URLs."
    )
    return df, source


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """Return dict of name -> sklearn Pipeline."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=10, min_samples_split=5,
                random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=5, n_jobs=-1,
                random_state=RANDOM_STATE
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(5), scoring="f1"
    )

    metrics = {
        "name":       name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "auc":        roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std":  cv_scores.std(),
        "train_time": train_time,
        "y_pred":     y_pred,
        "y_proba":    y_proba,
        "model":      model,
        "clf_report": classification_report(
            y_test, y_pred,
            target_names=["Phishing", "Legitimate"],
            zero_division=0,
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, results: list, outpath: str):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Phishing", "Legitimate"],
            yticklabels=["Phishing", "Legitimate"],
            ax=ax,
        )
        ax.set_title(res["name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"[OK] Saved confusion matrix -> {outpath}")


def plot_roc_curves(y_test, results: list, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for res in results:
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, label=f"{res['name']} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"[OK] Saved ROC curve    -> {outpath}")


def plot_feature_importance(best_result: dict, feature_names: list, outpath: str):
    model = best_result["model"]
    # Try to get feature importances from the final estimator
    clf = model[-1] if hasattr(model, "__len__") else model
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("[INFO] Model does not expose feature importances; skipping plot.")
        return

    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if i < 5 else "#2980b9" for i in range(len(importances))]
    ax.bar(range(len(importances)), importances[idx], color=colors)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9
    )
    ax.set_title(
        f"Feature Importances — {best_result['name']}", fontweight="bold"
    )
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"[OK] Saved feature imp. -> {outpath}")


def plot_model_comparison(results: list, outpath: str):
    metrics = ["accuracy", "precision", "recall", "f1"]
    names   = [r["name"] for r in results]
    x       = np.arange(len(names))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in results]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"[OK] Saved model comp.  -> {outpath}")


# ---------------------------------------------------------------------------
# 5. Save evaluation report
# ---------------------------------------------------------------------------

def save_text_report(results: list, best: dict, source: str, outpath: str):
    lines = [
        "=" * 70,
        "  PHISHING WEBSITE DETECTION — MODEL EVALUATION REPORT",
        "=" * 70,
        f"Dataset source : {source}",
        "",
    ]
    for res in results:
        lines += [
            f"Model : {res['name']}",
            f"  Accuracy   : {res['accuracy']:.4f}",
            f"  Precision  : {res['precision']:.4f}",
            f"  Recall     : {res['recall']:.4f}",
            f"  F1-Score   : {res['f1']:.4f}",
            f"  AUC-ROC    : {res['auc']:.4f}" if res['auc'] else "",
            f"  CV F1 (5-fold): {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}",
            f"  Train time : {res['train_time']:.2f}s",
            "",
            "  Classification Report:",
            res["clf_report"],
            "-" * 70,
        ]
    lines += [
        f"Best Model: {best['name']}  (F1 = {best['f1']:.4f})",
        "=" * 70,
    ]
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[OK] Saved text report  -> {outpath}")


# ---------------------------------------------------------------------------
# 6. Main training pipeline
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Phishing Detection — Training Pipeline")
    print("=" * 60)

    # -- Load data --
    df, source = load_dataset()
    print(f"[INFO] Dataset size  : {len(df):,} rows")
    print(f"[INFO] Dataset source: {source}")

    label_col  = "label"
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(float)
    y = df[label_col].values.astype(int)

    print(f"[INFO] Features      : {len(feature_cols)}")
    print(f"[INFO] Class balance : Phishing={int((y==0).sum())}, Legit={int((y==1).sum())}")

    # -- Train/test split --
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train={len(X_train)}, Test={len(X_test)}")

    # -- Train & evaluate all models --
    models  = build_models()
    results = []
    for name, model in models.items():
        print(f"\n[TRAIN] {name}...")
        res = evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(res)
        print(
            f"       Accuracy={res['accuracy']:.4f} "
            f"F1={res['f1']:.4f} "
            f"AUC={res['auc']:.4f}" if res['auc'] else "AUC=N/A"
        )

    # -- Select best model (highest F1) --
    best = max(results, key=lambda r: r["f1"])
    print(f"\n[BEST] {best['name']}  (F1={best['f1']:.4f})")

    # -- Save model + metadata --
    metadata = {
        "model_name":    best["name"],
        "feature_names": feature_cols,
        "accuracy":      best["accuracy"],
        "precision":     best["precision"],
        "recall":        best["recall"],
        "f1":            best["f1"],
        "auc":           best["auc"],
        "dataset_source": source,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
    }
    joblib.dump({"model": best["model"], "metadata": metadata, "feature_names": feature_cols},
                MODEL_PATH)
    print(f"[OK] Model saved     -> {MODEL_PATH}")

    # Also save a plain copy to project root for convenience
    root_model = os.path.join(BASE_DIR, "model.pkl")
    joblib.dump({"model": best["model"], "metadata": metadata, "feature_names": feature_cols},
                root_model)
    print(f"[OK] Model saved     -> {root_model}")

    # Save metadata JSON
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump({k: v for k, v in metadata.items()
                   if not isinstance(v, (np.ndarray,))}, fh, indent=2)

    # -- Generate reports and plots --
    plot_confusion_matrix(
        y_test, results,
        os.path.join(REPORTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_test, results,
        os.path.join(REPORTS_DIR, "roc_curve.png")
    )
    plot_feature_importance(
        best, feature_cols,
        os.path.join(REPORTS_DIR, "feature_importance.png")
    )
    plot_model_comparison(
        results,
        os.path.join(REPORTS_DIR, "model_comparison.png")
    )
    save_text_report(
        results, best, source,
        os.path.join(REPORTS_DIR, "evaluation.txt")
    )

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model : {best['name']}")
    print(f"  Accuracy   : {best['accuracy']:.4f}")
    print(f"  F1-Score   : {best['f1']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
