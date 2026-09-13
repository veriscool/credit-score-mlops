"""Train candidate credit-score models, compare, and package the winner.

Saves the winning model as a self-contained CreditScoreBundle
(artifacts/best_model.pkl) and packages it as model_artifact/model.tar.gz,
ready to upload to S3 for SageMaker deployment.

Preprocessing/splitting, model definitions, and evaluation each live in their
own file under src/ (data.py, models.py, evaluate.py, bundle.py) -- this
script is only the orchestrator.

Run from a SageMaker Notebook Instance:
    pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn optuna
    python pipeline.py --data data_A.csv --n-trials 0
"""

import argparse
import json
import logging
import os
import sys
import tarfile

sys.path.insert(0, "src")
from data import CreditDataPreprocessor, DataSplitter, LABEL_NAMES, load_dataset  # noqa: E402
from models import build_models  # noqa: E402
from evaluate import ModelEvaluator, print_comparison, select_best  # noqa: E402
from bundle import CreditScoreBundle  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ARTIFACT_DIR = "artifacts"
MODEL_ARTIFACT_DIR = "model_artifact"
TARBALL_PATH = os.path.join(MODEL_ARTIFACT_DIR, "model.tar.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data_A.csv", help="Path to raw CSV.")
    p.add_argument("--n-trials", type=int, default=0,
                    help="Optuna trials per model (0 = tuned defaults, fast).")
    p.add_argument(
        "--models", nargs="+", default=None, metavar="NAME",
        help=("One or more model names to train (default: all six). "
              "Choices: xgboost random_forest lightgbm knn extra_trees decision_tree"),
    )
    p.add_argument("--no-smote", action="store_true",
                    help="Disable SMOTE oversampling (enabled by default).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)

    print("Loading dataset...")
    raw = load_dataset(args.data)
    print(f"Dataset shape: {raw.shape}")

    preprocessor = CreditDataPreprocessor()
    clean = preprocessor.fit_transform(raw)
    preprocessor.save(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(preprocessor.feature_columns_, f, indent=2)

    splitter = DataSplitter(random_state=args.seed)
    X_train, X_test, y_train, y_test = splitter.split(clean)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Class distribution (train): " + ", ".join(
        f"{LABEL_NAMES[c]}={int((y_train == c).sum())}" for c in sorted(y_train.unique())
    ))

    groups_train = DataSplitter.train_groups(clean, splitter)

    smote = None
    if not args.no_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=args.seed)
        print("SMOTE enabled (imblearn, k_neighbors=5, strategy=auto)")

    all_models = build_models(args.seed)
    if args.models:
        unknown = set(args.models) - set(all_models)
        if unknown:
            sys.exit(f"Unknown model(s): {unknown}. Valid names: {', '.join(all_models)}")
        candidates = {name: all_models[name] for name in args.models}
    else:
        candidates = all_models

    evaluator = ModelEvaluator()
    results = {}
    trained_models = {}

    for name, model_def in candidates.items():
        print(f"\nTraining {name}...")
        params = model_def.tune(X_train, y_train, groups_train, args.n_trials, smote=smote)
        result = model_def.fit_final(X_train, y_train, params, smote=smote, groups=groups_train)
        results[name] = evaluator.evaluate(result.model, X_test, y_test, name, ARTIFACT_DIR)
        trained_models[name] = result.model

    print_comparison(results)

    best_name = select_best(results)
    best_model = trained_models[best_name]
    print(f"\nWinner: {best_name}  (macro-F1={results[best_name].macro_f1:.4f})")

    bundle = CreditScoreBundle(
        preprocessor=preprocessor,
        model=best_model,
        feature_columns=preprocessor.feature_columns_,
        label_names=LABEL_NAMES,
        model_name=best_name,
    )
    bundle_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    bundle.save(bundle_path)
    with open(os.path.join(ARTIFACT_DIR, "best_model.json"), "w") as f:
        json.dump({"name": best_name, "macro_f1": results[best_name].macro_f1}, f, indent=2)

    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(bundle_path, arcname="best_model.pkl")
    print(f"Packaged: {TARBALL_PATH}")

    print("\nNext steps:")
    print("  Make the s3 bucket with:\n\n  aws s3 mb s3://your-bucket-name --region us-east-1\n\n  and upload with:")
    print(f"\n  aws s3 cp {TARBALL_PATH} s3://your-bucket-name/credit-score/model.tar.gz\n")


if __name__ == "__main__":
    main()
