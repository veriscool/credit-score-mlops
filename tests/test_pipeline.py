"""
Smoke tests for the credit-score pipeline: preprocessor output shape and
end-to-end bundle prediction on the self-test records.

Run from the repo root:
    pytest tests/
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.pkl"
BUNDLE_PATH = ROOT / "artifacts" / "best_model.pkl"
FEATURE_COLUMNS_PATH = ROOT / "artifacts" / "feature_columns.json"

pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH.exists() or not PREPROCESSOR_PATH.exists(),
    reason="artifacts/ not present - run pipeline.py to train first",
)


class _PipelineUnpickler(pickle.Unpickler):
    """Bundles were pickled while pipeline.py ran as __main__; remap that
    back to the real module so they unpickle from any entry point."""

    _REMAP = {"CreditScoreBundle", "CreditDataPreprocessor"}

    def find_class(self, module, name):
        if module == "__main__" and name in self._REMAP:
            import pipeline
            return getattr(pipeline, name)
        return super().find_class(module, name)


def _load(path: Path):
    with open(path, "rb") as f:
        return _PipelineUnpickler(f).load()


@pytest.fixture(scope="module")
def bundle():
    return _load(BUNDLE_PATH)


@pytest.fixture(scope="module")
def selftest_records():
    from predict import SELFTEST_RECORDS
    return SELFTEST_RECORDS


def test_feature_columns_count():
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    assert len(feature_columns) == 46


def test_preprocessor_output_shape(selftest_records):
    import pandas as pd

    # Customer_ID survives preprocessor.transform() (kept for leakage-safe
    # grouping) but is dropped afterwards via the ordered feature_columns
    # list, same as CreditScoreBundle._features() does.
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    preprocessor = _load(PREPROCESSOR_PATH)
    clean = [{k: v for k, v in r.items() if k != "_expected"} for r in selftest_records]
    out = preprocessor.transform(pd.DataFrame(clean))
    assert out[feature_columns].shape[1] == 46


def test_bundle_predicts_selftest_records(bundle, selftest_records):
    import pandas as pd

    clean = [{k: v for k, v in r.items() if k != "_expected"} for r in selftest_records]
    expected = [r["_expected"] for r in selftest_records]

    df = pd.DataFrame(clean)
    predicted = bundle.predict_labels(df)

    assert predicted == expected


def test_bundle_predict_proba_sums_to_one(bundle, selftest_records):
    import numpy as np
    import pandas as pd

    clean = [{k: v for k, v in r.items() if k != "_expected"} for r in selftest_records]
    probas = bundle.predict_proba(pd.DataFrame(clean))

    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)
