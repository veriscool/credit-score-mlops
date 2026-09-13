"""SageMaker inference entry point for the Credit Score model.

Generic loader for the project's self-contained CreditScoreBundle (fitted
preprocessor + trained model + ordered feature list + label names). Works for
whichever model won (champion is currently LightGBM, see
../../artifacts/best_model.json) without any changes here.

Because pipeline.py trains by importing CreditScoreBundle from bundle.py and
CreditDataPreprocessor from data.py (rather than defining them inline in a
script that runs as __main__), the pickle references those real module names
directly -- a plain pickle.load() resolves them as long as this directory is
on sys.path, which SageMaker guarantees via source_dir="src". This is why,
unlike the project's local ../../scripts/predict.py / ../../app/streamlit_app.py,
no custom __main__-remapping Unpickler is needed here.

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""

import json
import os
import pickle

import pandas as pd

# Importing these registers "data" / "bundle" as already-loaded modules
# before pickle.load() needs to resolve them.
from bundle import CreditScoreBundle  # noqa: F401
from data import CreditDataPreprocessor  # noqa: F401


JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir: str):
    """Load the CreditScoreBundle pickled at model_dir/best_model.pkl."""
    bundle_path = os.path.join(model_dir, "best_model.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # The champion may have been trained on GPU; force CPU so a GPU-less
    # inference instance doesn't crash. No-op for models without this param
    # (e.g. LightGBM's sklearn API does not expose device via set_params).
    if hasattr(bundle.model, "set_params"):
        try:
            bundle.model.set_params(device="cpu")
        except Exception:
            pass
    return bundle


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame of *raw* credit records.

    Accepts JSON only: {"instances": [ {..23 raw fields..}, ... ]}
    A single record may be sent as a bare object instead of a one-item list:
    {"instances": {..raw fields..}}

    Records are raw mixed-type dicts (strings for Month/Occupation/Credit_Mix/
    etc, numbers for the rest) because CreditDataPreprocessor -- bundled
    inside the model -- expects the same 23 raw CSV columns the local
    Streamlit app and predict.py CLI send, not a flat numeric feature vector.
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        if isinstance(instances, dict):
            instances = [instances]
        return pd.DataFrame(instances)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, bundle: "CreditScoreBundle") -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    probs = bundle.predict_proba(input_data)
    labels = bundle.predict_labels(input_data)
    class_ids = [bundle.label_names.index(label) for label in labels]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids,
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
