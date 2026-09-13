"""Evaluate trained models and produce a comparison table.

Metrics, comparison table, and winner selection, extracted from the
project's local ../../pipeline.py. Primary metric is macro-F1 (classes are
imbalanced, minority class ~17-19%), matching the local training pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data import LABEL_NAMES

logger = logging.getLogger("evaluate")


@dataclass
class EvalResult:
    macro_f1: float
    accuracy: float
    report: Dict[str, Any]
    confusion_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class ModelEvaluator:
    """Computes metrics and renders a confusion-matrix image artifact."""

    def __init__(self, label_names: List[str] = LABEL_NAMES) -> None:
        self.label_names = label_names

    def evaluate(self, model, X_test, y_test, name: str, artifact_dir: str) -> EvalResult:
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        y_pred = model.predict(X_test)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test, y_pred, target_names=self.label_names, output_dict=True)

        metrics = {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "accuracy": acc}
        for lbl in self.label_names:
            metrics[f"f1_{lbl}"] = float(report[lbl]["f1-score"])
            metrics[f"recall_{lbl}"] = float(report[lbl]["recall"])

        cm_path = self._plot_confusion(y_test, y_pred, name, artifact_dir)
        logger.info("[%s] test macro-F1=%.4f | accuracy=%.4f", name, macro_f1, acc)
        return EvalResult(macro_f1, acc, report, cm_path, metrics)

    def _plot_confusion(self, y_true, y_pred, name: str, artifact_dir: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        os.makedirs(artifact_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=self.label_names, ax=ax, colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix (Test)")
        fig.tight_layout()
        path = os.path.join(artifact_dir, f"confusion_{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


def print_comparison(results: Dict[str, EvalResult]) -> None:
    """Print a comparison table across models. results = {name: EvalResult}."""
    print(f"\n{'Model':<16} {'Macro F1':>10} {'Accuracy':>10}")
    print("-" * 40)
    for name, ev in sorted(results.items(), key=lambda kv: kv[1].macro_f1, reverse=True):
        print(f"{name:<16} {ev.macro_f1:>10.4f} {ev.accuracy:>10.4f}")


def select_best(results: Dict[str, EvalResult]) -> str:
    """Return the name of the model with the highest macro-F1."""
    return max(results, key=lambda name: results[name].macro_f1)
