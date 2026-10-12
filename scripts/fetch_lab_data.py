"""Fetch a small catalogue of real tabular datasets for the scientist's lab.

    var/lab-venv/bin/python scripts/fetch_lab_data.py [--into var/lab-data]

Experiments run without network, so real data has to be on disk before they start.
Each dataset is saved as ``<name>.npz`` holding ``X`` (float64, categorical columns as
integer codes, missing values as NaN), ``y`` (integer class codes, or float for
regression), ``feature_names``, ``categorical`` (a boolean mask) and ``classes``, and
described in ``manifest.json``. The lab mounts the folder read-only and tells whoever
designs an experiment what is in it. Nothing here is specific to any one question.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (name, OpenML id, task, one line about it). Small on purpose: the whole set is a few MB.
CATALOGUE = (
    ("adult", 1590, "classification", "census income >50K; mixed numeric and categorical; missing values"),
    ("bank_marketing", 1461, "classification", "term-deposit subscription; imbalanced (~12% positive)"),
    ("credit_g", 31, "classification", "German credit risk; 1,000 rows; mostly categorical"),
    ("electricity", 151, "classification", "price direction in the NSW market; ordered in time"),
    ("phoneme", 1489, "classification", "nasal vs oral vowels; 5 numeric features; imbalanced"),
    ("spambase", 44, "classification", "spam e-mail; 57 word/character frequencies"),
    ("kc1", 1067, "classification", "software defect prediction; imbalanced"),
    ("blood_transfusion", 1464, "classification", "blood donation; 748 rows, 4 features"),
    ("diabetes_pima", 37, "classification", "Pima diabetes onset; 768 rows"),
    ("segment", 40984, "classification", "image segment type; 7 balanced classes"),
    ("vehicle", 54, "classification", "vehicle silhouette type; 4 classes"),
    ("satimage", 182, "classification", "Landsat soil type; 6 classes"),
    ("letter", 6, "classification", "capital letter from pixel statistics; 26 classes, 20,000 rows"),
    ("california_housing", 43939, "regression", "median house value by block group; 20,640 rows"),
)


def fetch(name: str, data_id: int, task: str, cache: Path) -> dict:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff", data_home=str(cache))
    X = np.asarray(bunch.data, dtype=np.float64)
    names = [str(n) for n in bunch.feature_names]
    categories = dict(bunch.categories or {})
    categorical = np.array([n in categories for n in names])
    target = np.asarray(bunch.target)
    if task == "classification":
        classes, y = np.unique(target.astype(str), return_inverse=True)
        y = y.astype(np.int64)
    else:
        classes, y = np.array([]), target.astype(np.float64)
    return {"X": X, "y": y, "feature_names": np.array(names), "categorical": categorical,
            "classes": np.array([str(c) for c in classes])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", type=Path, default=ROOT / "var" / "lab-data")
    args = parser.parse_args(argv)
    args.into.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "npz with X, y, feature_names, categorical, classes", "datasets": {}}
    cache = Path(tempfile.mkdtemp(prefix="openml-", dir=args.into))
    try:
        for name, data_id, task, about in CATALOGUE:
            try:
                arrays = fetch(name, data_id, task, cache)
            except Exception as exc:  # one unreachable dataset should not cost the others
                print(f"skip {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            np.savez_compressed(args.into / f"{name}.npz", **arrays)
            X, y = arrays["X"], arrays["y"]
            manifest["datasets"][name] = {
                "file": f"{name}.npz", "task": task, "rows": int(X.shape[0]), "features": int(X.shape[1]),
                "categorical_features": int(arrays["categorical"].sum()),
                "missing_values": bool(np.isnan(X).any()),
                "classes": len(arrays["classes"]) if task == "classification" else None,
                "about": about, "source": f"OpenML data id {data_id}"}
            print(f"{name}: {X.shape}", flush=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    (args.into / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
