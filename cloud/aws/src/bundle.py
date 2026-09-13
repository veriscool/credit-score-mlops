"""Self-contained scoring bundle for a trained credit-score model.

Bundles everything needed to score a *raw* credit record in one object:
the fitted preprocessor, the trained model, the ordered feature list, and
the label names. Extracted from ../../pipeline.py's CreditScoreBundle.

Because pipeline.py (this folder's root orchestrator) imports this class
from its own module -- rather than defining it inline in a script that runs
as __main__ -- the pickled bundle references "bundle.CreditScoreBundle" and
"data.CreditDataPreprocessor" directly. That means src/inference.py can load
it with a plain pickle.load(); no __main__-remapping unpickler (as used by
the local ../../scripts/predict.py / ../../app/streamlit_app.py) is needed here.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, List, Optional

import pandas as pd

from data import LABEL_NAMES

logger = logging.getLogger("bundle")


class CreditScoreBundle:
    def __init__(self, preprocessor, model: Any, feature_columns: List[str],
                 label_names: Optional[List[str]] = None, model_name: str = "") -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.feature_columns = list(feature_columns)
        self.label_names = list(label_names) if label_names is not None else list(LABEL_NAMES)
        self.model_name = model_name

    def _features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        out = self.preprocessor.transform(raw_df)
        return out[self.feature_columns]

    def predict(self, raw_df: pd.DataFrame):
        # .to_numpy(): xgboost's DataFrame predict path imports pandas'
        # Float32Dtype/Float64Dtype (pandas>=1.2), which the inference
        # container's pinned pandas 1.1.3 doesn't have. Array input skips it.
        return self.model.predict(self._features(raw_df).to_numpy())

    def predict_proba(self, raw_df: pd.DataFrame):
        return self.model.predict_proba(self._features(raw_df).to_numpy())

    def predict_labels(self, raw_df: pd.DataFrame) -> List[str]:
        return [self.label_names[int(i)] for i in self.predict(raw_df)]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Bundle saved -> %s", path)

    @staticmethod
    def load(path: str) -> "CreditScoreBundle":
        with open(path, "rb") as f:
            return pickle.load(f)
