#!/usr/bin/env python3
"""
map_quality_classifier.py
─────────────────────────────────────────────────────────────────────────────
Runtime consumer of the trained arena-map quality model (sweep iteration v3, a
scikit-learn RandomForest). Feeds the diagnostic feature vector that the
BuildArenaMap action returns — the dict the mission orchestrator stores as
``self._map_features`` — into the model and answers one question:

    is the stitched arena map good enough to use?  → bool

Model artifacts (written by sweep/stitching/train_classifier.ipynb,
                 then made portable by sweep/stitching/export_portable_model.py)
───────────────────────────────────────────────────────────────────
    models/latest/                (symlink → models/v3)
        forest.npz                numpy-only RandomForest (preferred at runtime)
        portable_meta.json        n_trees / pass_class_index for forest.npz
        model.joblib              original sklearn estimator (fallback only)
        feature_names.json        the 45 ordered feature columns used to train
        threshold.json            decision threshold + training metadata
                                  → predict "good" when P(pass) >= threshold

Feature alignment
─────────────────
``MapDiagnostics.to_feature_vector()`` produces 46 features, but the training
notebook selected columns by group prefix and dropped ``inter_marker_distance_norm``
(it has no group prefix), leaving 45. Keying strictly off feature_names.json
makes this automatic: extra keys in the input dict are ignored and the column
order always matches what the model was fit on. Any expected feature missing
from the input is filled with the training sentinel (-1.0).

Dependencies (inference host)
─────────────────────────────
numpy ONLY, when forest.npz is present (the normal Jetson path). The forest is
traversed in pure numpy, so no scikit-learn / joblib / onnxruntime is required.
If forest.npz is absent the class falls back to loading model.joblib, which then
does require joblib + a matching scikit-learn (1.9.0) — intended only for hosts
that have the training env. Generate forest.npz with export_portable_model.py.

Usage — library
───────────────
    from mission_orchestrator.map_quality_classifier import MapQualityClassifier
    clf = MapQualityClassifier()                  # loads models/latest
    result = clf.evaluate(features)               # features: Dict[str, float]
    if result.good:
        ...
    # or, just the boolean:
    if clf.is_map_good(features):
        ...

Usage — CLI (for testing with a captured feature dict)
──────────────────────────────────────────────────────
    python3 map_quality_classifier.py features.json
    python3 map_quality_classifier.py --model-dir /path/to/models/v3 features.json
    cat features.json | python3 map_quality_classifier.py -
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Default model directory. Overridable via the MAP_QUALITY_MODEL_DIR env var or
# the constructor argument. Points at the sweep's versioned model store, whose
# `latest` symlink tracks the newest trained version (currently v3).
_DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "MAP_QUALITY_MODEL_DIR",
        "/home/jetson/collab_nav_ground-jetson/src/arena_map_builder/"
        "sweep/stitching/models/latest",
    )
)

# Sentinel for any feature the model expects but the input dict lacks. Matches
# the training convention (None / unavailable → -1.0). An all-missing input
# therefore scores as a uniformly degraded map, which the model treats as bad —
# a safe default when diagnostics could not be computed upstream.
_MISSING_SENTINEL = -1.0


def build_feature_row(
    features: Dict[str, float],
    feature_names: Sequence[str],
    missing_value: float = _MISSING_SENTINEL,
) -> Tuple[np.ndarray, List[str]]:
    """Order `features` into a single model-input row per `feature_names`.

    Returns (row_2d, missing_keys):
      • row_2d        shape (1, len(feature_names)), float — ready for predict
      • missing_keys  feature names not present in `features` (filled with the
                      sentinel); empty when the input is complete.
    """
    missing = [k for k in feature_names if k not in features]
    row = [float(features.get(k, missing_value)) for k in feature_names]
    return np.asarray([row], dtype=float), missing


class _PortableForest:
    """Pure-numpy RandomForest scorer loaded from forest.npz.

    Reproduces sklearn's RandomForestClassifier.predict_proba: average, over all
    trees, of each tree's normalized leaf distribution. Node indices are local to
    each tree (sliced via `offsets`); leaves have children_left == -1.
    """

    def __init__(self, npz_path: Path, meta: dict):
        data = np.load(npz_path)
        self._cl = data["children_left"]
        self._cr = data["children_right"]
        self._feat = data["feature"]
        self._thr = data["threshold"]
        self._val = data["value"]
        self._off = data["offsets"]
        self.n_trees = int(meta["n_trees"])
        self.pass_class_index = int(meta["pass_class_index"])
        self.n_features = int(meta["n_features"])

    def predict_pass_proba(self, x_2d: np.ndarray) -> np.ndarray:
        """P(pass) for each row of x_2d (shape (n, n_features))."""
        n = x_2d.shape[0]
        acc = np.zeros((n, self._val.shape[1]), dtype=np.float64)
        for i in range(n):
            x = x_2d[i]
            row = np.zeros(self._val.shape[1], dtype=np.float64)
            for t in range(self.n_trees):
                a, b = self._off[t], self._off[t + 1]
                cl, cr, feat, thr = self._cl[a:b], self._cr[a:b], self._feat[a:b], self._thr[a:b]
                node = 0
                while cl[node] != -1:               # internal node
                    node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
                row += self._val[a + node]
            acc[i] = row / self.n_trees
        return acc[:, self.pass_class_index]


class _JoblibForest:
    """Fallback scorer: the original sklearn estimator via joblib.

    Only usable on hosts that have joblib + a compatible scikit-learn. Prefer the
    portable forest.npz on the Jetson.
    """

    def __init__(self, model_path: Path):
        import joblib
        self._model = joblib.load(model_path)
        # classes_ is [0, 1] → column 1 is P(pass); resolve defensively.
        self.pass_class_index = int(np.where(np.asarray(self._model.classes_) == 1)[0][0])

    def predict_pass_proba(self, x_2d: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x_2d)[:, self.pass_class_index]


@dataclass
class MapQualityResult:
    """Outcome of one map-quality evaluation."""
    good: bool            # p_pass >= threshold
    p_pass: float         # model's P(map is acceptable) in [0, 1]
    threshold: float      # decision threshold the model was tuned to
    missing: List[str]    # expected features absent from the input dict

    def __str__(self) -> str:
        verdict = "GOOD" if self.good else "BAD"
        warn = f"  [!] {len(self.missing)} missing feat" if self.missing else ""
        return (f"map quality: {verdict}  "
                f"P(pass)={self.p_pass:.3f}  thr={self.threshold:.3f}{warn}")


class MapQualityClassifier:
    """Loads the trained model and scores diagnostic feature vectors."""

    def __init__(self, model_dir: Optional[Union[Path, str]] = None):
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        feats_path = self.model_dir / "feature_names.json"
        meta_path = self.model_dir / "threshold.json"
        for p in (feats_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(f"model artifact missing: {p}")

        self.feature_names: List[str] = json.loads(feats_path.read_text())
        meta = json.loads(meta_path.read_text())
        self.threshold = float(meta["threshold"])
        self.model_type = meta.get("model_type", "?")
        self.version = meta.get("version", "?")

        # Prefer the numpy-only portable forest (no scikit-learn needed). Fall
        # back to the pickled sklearn estimator only if it is absent.
        npz_path = self.model_dir / "forest.npz"
        pmeta_path = self.model_dir / "portable_meta.json"
        model_path = self.model_dir / "model.joblib"
        if npz_path.is_file() and pmeta_path.is_file():
            self._scorer = _PortableForest(
                npz_path, json.loads(pmeta_path.read_text()))
            self.backend = "portable"
            if self._scorer.n_features != len(self.feature_names):
                raise ValueError(
                    f"forest.npz expects {self._scorer.n_features} features but "
                    f"feature_names.json lists {len(self.feature_names)}")
        elif model_path.is_file():
            self._scorer = _JoblibForest(model_path)
            self.backend = "joblib"
        else:
            raise FileNotFoundError(
                f"no usable model in {self.model_dir}: need forest.npz "
                f"(+ portable_meta.json) or model.joblib")

    # ── public API ─────────────────────────────────────────────────────────

    def evaluate(self, features: Dict[str, float]) -> MapQualityResult:
        """Score `features` and return the full result (verdict + probability)."""
        x, missing = build_feature_row(features, self.feature_names)
        p_pass = float(self._scorer.predict_pass_proba(x)[0])
        return MapQualityResult(
            good=p_pass >= self.threshold,
            p_pass=p_pass,
            threshold=self.threshold,
            missing=missing,
        )

    def is_map_good(self, features: Dict[str, float]) -> bool:
        """Return just the boolean verdict: is the map good enough to use?"""
        return self.evaluate(features).good

    def __repr__(self) -> str:
        return (f"MapQualityClassifier(model={self.model_type} v{self.version}, "
                f"backend={self.backend}, threshold={self.threshold:.4f}, "
                f"n_features={len(self.feature_names)}, dir={self.model_dir})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — score a JSON feature dict from a file or stdin (for manual testing)
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify an arena map as good/bad from its diagnostic "
                    "feature vector (JSON dict of feature_name -> value).")
    parser.add_argument(
        "features", help="Path to a JSON file with the feature dict, or '-' for stdin")
    parser.add_argument(
        "--model-dir", default=None,
        help="Model directory (default: $MAP_QUALITY_MODEL_DIR or the sweep "
             "models/latest)")
    args = parser.parse_args()

    import sys
    raw = sys.stdin.read() if args.features == "-" else Path(args.features).read_text()
    features = json.loads(raw)
    if not isinstance(features, dict):
        print("[error] input JSON must be an object of feature_name -> value",
              file=sys.stderr)
        return 2

    clf = MapQualityClassifier(args.model_dir)
    result = clf.evaluate(features)
    print(repr(clf))
    print(result)
    if result.missing:
        print(f"  missing features filled with {_MISSING_SENTINEL}: "
              f"{', '.join(result.missing)}", file=sys.stderr)
    # Exit code doubles as the verdict: 0 = good map, 1 = bad map.
    return 0 if result.good else 1


if __name__ == "__main__":
    raise SystemExit(_main())
