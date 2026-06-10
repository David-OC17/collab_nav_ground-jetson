#!/usr/bin/env python3
"""
export_portable_model.py
─────────────────────────────────────────────────────────────────────────────
Convert a trained scikit-learn RandomForest (models/v{N}/model.joblib) into a
dependency-light, numpy-only representation (forest.npz) that the Jetson can
evaluate without scikit-learn, joblib, or onnxruntime installed.

WHY
───
The model is trained on a laptop (scikit-learn 1.9.0) but consumed on the Jetson
inside the ROS Python, which has only numpy. A pickled sklearn estimator can't be
loaded there. A RandomForest is just an ensemble of decision trees, and every
tree's structure is exposed via `tree_.children_left/right`, `tree_.feature`,
`tree_.threshold`, and `tree_.value`. Dumping those arrays lets us reproduce
`predict_proba` exactly with plain numpy tree traversal.

This script MUST run in the training environment (where scikit-learn + joblib
can read model.joblib) — e.g. the laptop mamba env. It writes forest.npz +
portable_meta.json next to model.joblib and verifies the numpy implementation
matches sklearn's predict_proba to within a tiny tolerance before saving.

USAGE
─────
    python3 export_portable_model.py                       # models/latest
    python3 export_portable_model.py --model-dir models/v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _collect_forest(model):
    """Flatten every tree's arrays into concatenated numpy arrays + offsets.

    Node indices in children_left/right are LOCAL to each tree; we keep them
    local and slice per tree at inference, so no index rewriting is needed.
    Leaf nodes have children_left == children_right == -1 (sklearn TREE_LEAF).

    `value` is normalized per node to a class-probability distribution — exactly
    what each tree contributes to RandomForest.predict_proba (this correctly
    reproduces class_weight='balanced_subsample', whose effect is already baked
    into the weighted counts stored in tree_.value).
    """
    estimators = model.estimators_
    classes = np.asarray(model.classes_)
    n_classes = len(classes)

    children_left, children_right, feature, threshold, value = [], [], [], [], []
    offsets = [0]

    for est in estimators:
        t = est.tree_
        children_left.append(t.children_left.astype(np.int32))
        children_right.append(t.children_right.astype(np.int32))
        feature.append(t.feature.astype(np.int32))
        threshold.append(t.threshold.astype(np.float64))
        # t.value: (n_nodes, n_outputs, n_classes); single-output → [:, 0, :]
        v = t.value[:, 0, :].astype(np.float64)
        row_sums = v.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0           # guard (shouldn't happen)
        value.append(v / row_sums)
        offsets.append(offsets[-1] + t.node_count)

    arrays = {
        "children_left":  np.concatenate(children_left),
        "children_right": np.concatenate(children_right),
        "feature":        np.concatenate(feature),
        "threshold":      np.concatenate(threshold),
        "value":          np.concatenate(value, axis=0),
        "offsets":        np.asarray(offsets, dtype=np.int64),
    }
    pass_class_index = int(np.where(classes == 1)[0][0])  # P(pass) column
    return arrays, n_classes, pass_class_index, len(estimators)


def portable_predict_proba(arrays, n_trees, x_2d):
    """Numpy-only RandomForest predict_proba — mirrors the runtime consumer.

    Kept here too so the export can self-verify against sklearn before saving.
    """
    CL, CR = arrays["children_left"], arrays["children_right"]
    FEAT, THR = arrays["feature"], arrays["threshold"]
    VAL, OFF = arrays["value"], arrays["offsets"]

    out = np.zeros((x_2d.shape[0], VAL.shape[1]), dtype=np.float64)
    for i in range(x_2d.shape[0]):
        x = x_2d[i]
        acc = np.zeros(VAL.shape[1], dtype=np.float64)
        for t in range(n_trees):
            a, b = OFF[t], OFF[t + 1]
            cl, cr, feat, thr = CL[a:b], CR[a:b], FEAT[a:b], THR[a:b]
            node = 0
            while cl[node] != -1:                 # not a leaf
                if x[feat[node]] <= thr[node]:
                    node = cl[node]
                else:
                    node = cr[node]
            acc += VAL[a + node]
        out[i] = acc / n_trees
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="models/latest",
                        help="Directory holding model.joblib (default: models/latest)")
    parser.add_argument("--n-check", type=int, default=2000,
                        help="Random rows used to verify parity with sklearn")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="Max allowed |portable - sklearn| on P(pass)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    model_path = model_dir / "model.joblib"
    feats_path = model_dir / "feature_names.json"
    if not model_path.is_file():
        print(f"[error] {model_path} not found", file=sys.stderr)
        return 2

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        print(f"[error] model is {type(model).__name__}, not RandomForestClassifier. "
              f"This exporter only handles RandomForest.", file=sys.stderr)
        return 2

    feature_names = json.loads(feats_path.read_text()) if feats_path.is_file() else None
    n_features = model.n_features_in_
    if feature_names is not None and len(feature_names) != n_features:
        print(f"[error] feature_names.json has {len(feature_names)} entries but model "
              f"expects {n_features}", file=sys.stderr)
        return 2

    arrays, n_classes, pass_idx, n_trees = _collect_forest(model)
    print(f"  Forest: {n_trees} trees, {arrays['value'].shape[0]} total nodes, "
          f"{n_classes} classes, n_features={n_features}, pass_class_index={pass_idx}")

    # ── parity check against sklearn ─────────────────────────────────────────
    rng = np.random.default_rng(0)
    # Mix plausible value ranges with the -1.0 sentinel to exercise both paths.
    X = rng.uniform(-1.0, 3.0, size=(args.n_check, n_features))
    X[rng.random(X.shape) < 0.1] = -1.0
    ref = model.predict_proba(X)[:, pass_idx]
    got = portable_predict_proba(arrays, n_trees, X)[:, pass_idx]
    max_err = float(np.max(np.abs(ref - got)))
    print(f"  Parity check on {args.n_check} random rows: max |Δ P(pass)| = {max_err:.2e}")
    if max_err > args.tol:
        print(f"[error] parity check failed (> {args.tol:.0e}). Not writing forest.npz.",
              file=sys.stderr)
        return 1

    # ── save ─────────────────────────────────────────────────────────────────
    npz_path = model_dir / "forest.npz"
    np.savez_compressed(npz_path, n_trees=np.int64(n_trees),
                        pass_class_index=np.int64(pass_idx), **arrays)
    meta = {
        "format": "portable_random_forest_v1",
        "n_trees": n_trees,
        "n_classes": n_classes,
        "n_features": int(n_features),
        "pass_class_index": pass_idx,
        "source_model": str(model_path.name),
        "sklearn_class": type(model).__name__,
    }
    (model_dir / "portable_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Wrote {npz_path}  ({npz_path.stat().st_size/1024:.0f} KB)")
    print(f"  Wrote {model_dir / 'portable_meta.json'}")
    print("  Parity verified — the Jetson consumer can now run numpy-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
