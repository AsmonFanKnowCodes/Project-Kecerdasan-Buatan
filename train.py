"""Train and evaluate models for the synthetic HIV dataset."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 42
VIRAL_LOAD = "Current_Viral_Load"
FEATURES = [
    "Current_CD4_Count",
    VIRAL_LOAD,
    "TB_Coinfection",
    "Drug_Resistance_Mutation",
]
TARGET = "Mortality_5Yr_Outcome"
ID_COLUMN = "Patient_ID"


def read_dataset(path: Path) -> tuple[pd.DataFrame, int]:
    columns = [ID_COLUMN, *FEATURES, TARGET]
    column_count = len(pd.read_csv(path, nrows=0).columns)
    data = pd.read_csv(path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("Dataset harus berisi rekaman lengkap tanpa nilai kosong.")
    if not data[ID_COLUMN].is_unique:
        raise ValueError("Patient_ID harus unik agar rekaman tidak muncul lintas split.")
    if any(set(data[name].unique()) != {0, 1} for name in [*FEATURES[2:], TARGET]):
        raise ValueError("Kolom biner dan label target harus memuat nilai 0 dan 1.")
    if not data[FEATURES[0]].between(5, 1600).all() or not data[VIRAL_LOAD].between(0, 750000).all():
        raise ValueError("Nilai CD4 atau viral load berada di luar rentang generator dataset.")
    return data, column_count


def candidate_models() -> dict[str, Pipeline]:
    log_viral_load = ColumnTransformer(
        [("log_viral_load", FunctionTransformer(np.log1p), [VIRAL_LOAD])],
        remainder="passthrough",
    )
    return {
        "Logistic Regression": Pipeline([
            ("transform", log_viral_load),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=100, random_state=SEED, n_jobs=4,
            )),
        ]),
    }


def ranking_metrics(labels, scores) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def choose_threshold(labels, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # The final precision/recall point has no corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0)
    return float(thresholds[f1.argmax()])


def save_artifacts(model: Pipeline, report: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ARTIFACTS) as directory:
        model_path = Path(directory) / "model.joblib"
        report_path = Path(directory) / "metrics.json"
        joblib.dump(model, model_path, compress=3)
        report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        model_path.replace(ARTIFACTS / model_path.name)
        report_path.replace(ARTIFACTS / report_path.name)


def train(path: Path) -> None:
    data, column_count = read_dataset(path)
    x, y = data[FEATURES], data[TARGET]
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=SEED,
    )

    models = candidate_models()
    validation_rows = []
    validation_scores = {}
    for name, model in models.items():
        print(f"Melatih {name}...", flush=True)
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_validation)[:, 1]
        validation_scores[name] = scores
        validation_rows.append({"name": name, **ranking_metrics(y_validation, scores)})

    selected = max(validation_rows, key=lambda row: row["average_precision"])["name"]
    threshold = choose_threshold(y_validation, validation_scores[selected])
    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    dummy_scores = dummy.predict_proba(x_validation)[:, 1]
    validation_rows.append({"name": "Baseline (Dummy)", **ranking_metrics(y_validation, dummy_scores)})

    final_model = models[selected]
    test_scores = final_model.predict_proba(x_test)[:, 1]
    prediction = (test_scores >= threshold).astype(int)
    report = {
        "dataset": {"rows": len(data), "columns": column_count, "positive_rate": float(y.mean())},
        "split": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "seed": SEED,
        "target": TARGET,
        "features": FEATURES,
        "selected_model": selected,
        "threshold": threshold,
        "models": validation_rows,
        "test": {
            **ranking_metrics(y_test, test_scores),
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        },
    }
    save_artifacts(final_model, report)
    print(f"Model terpilih: {selected} | Average precision (uji): {report['test']['average_precision']:.4f}")
    print(f"Artefak disimpan di {ARTIFACTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "archive.zip", help="CSV atau ZIP berisi satu CSV.")
    args = parser.parse_args()
    try:
        train(args.data)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Training gagal: {error}\n")


if __name__ == "__main__":
    main()
