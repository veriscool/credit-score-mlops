"""
predict.py - Credit Score CLI inferencing script.

Loads artifacts/best_model.pkl (a CreditScoreBundle) and predicts on raw
credit records. Requires pipeline.py (at the repo root) to be importable.

Usage examples (run from the repo root)
----------------------------------------
# Self-test: one record per class (Poor / Standard / Good)
python scripts/predict.py --selftest

# Predict from an inline JSON object (single record)
python scripts/predict.py --json '{"Month":"March","Age":35,...}'

# Predict from a CSV file (one or more rows)
python scripts/predict.py --input records.csv

# Specify a different model bundle
python scripts/predict.py --selftest --bundle artifacts/best_model.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

# Locate the project root (this file lives in scripts/) so pipeline.py is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BUNDLE = ROOT / "artifacts" / "best_model.pkl"

# Self-test records: one representative raw record per expected class.
SELFTEST_RECORDS = [
    {
        # Poor: high delay, bad credit mix, many missed payments, high debt
        "Customer_ID": "CUS_TEST_001",
        "Month": "January",
        "Age": 28,
        "Annual_Income": 15000.0,
        "Monthly_Inhand_Salary": 1050.0,
        "Num_Bank_Accounts": 9,
        "Num_Credit_Card": 8,
        "Interest_Rate": 34,
        "Num_of_Loan": 7,
        "Type_of_Loan": "Payday Loan, and Personal Loan",
        "Delay_from_due_date": 62,
        "Num_of_Delayed_Payment": 22,
        "Changed_Credit_Limit": 0.5,
        "Num_Credit_Inquiries": 17.0,
        "Credit_Mix": "Bad",
        "Outstanding_Debt": 3500.0,
        "Credit_Utilization_Ratio": 42.0,
        "Credit_History_Age": "3 Years and 2 Months",
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 350.0,
        "Amount_invested_monthly": 10.0,
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Monthly_Balance": 50.0,
        "_expected": "Poor",
    },
    {
        # Standard: moderate profile
        "Customer_ID": "CUS_TEST_002",
        "Month": "June",
        "Age": 35,
        "Annual_Income": 48000.0,
        "Monthly_Inhand_Salary": 3600.0,
        "Num_Bank_Accounts": 4,
        "Num_Credit_Card": 4,
        "Interest_Rate": 14,
        "Num_of_Loan": 3,
        "Type_of_Loan": "Personal Loan, and Auto Loan",
        "Delay_from_due_date": 18,
        "Num_of_Delayed_Payment": 6,
        "Changed_Credit_Limit": 8.0,
        "Num_Credit_Inquiries": 5.0,
        "Credit_Mix": "Standard",
        "Outstanding_Debt": 1200.0,
        "Credit_Utilization_Ratio": 30.0,
        "Credit_History_Age": "10 Years and 4 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 120.0,
        "Amount_invested_monthly": 200.0,
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
        "Monthly_Balance": 350.0,
        "_expected": "Standard",
    },
    {
        # Good: low delay, good credit mix, low debt, high income
        "Customer_ID": "CUS_TEST_003",
        "Month": "September",
        "Age": 45,
        "Annual_Income": 120000.0,
        "Monthly_Inhand_Salary": 9500.0,
        "Num_Bank_Accounts": 2,
        "Num_Credit_Card": 3,
        "Interest_Rate": 7,
        "Num_of_Loan": 2,
        "Type_of_Loan": "Mortgage Loan, and Auto Loan",
        "Delay_from_due_date": 2,
        "Num_of_Delayed_Payment": 0,
        "Changed_Credit_Limit": 15.0,
        "Num_Credit_Inquiries": 1.0,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 200.0,
        "Credit_Utilization_Ratio": 18.0,
        "Credit_History_Age": "22 Years and 8 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 55.0,
        "Amount_invested_monthly": 900.0,
        "Payment_Behaviour": "Low_spent_Large_value_payments",
        "Monthly_Balance": 2500.0,
        "_expected": "Good",
    },
]


class _PipelineUnpickler(pickle.Unpickler):
    _REMAP = {"CreditScoreBundle", "CreditDataPreprocessor"}

    def find_class(self, module, name):
        if module == "__main__" and name in self._REMAP:
            import pipeline
            return getattr(pipeline, name)
        return super().find_class(module, name)


def load_bundle(bundle_path: str | Path):
    path = Path(bundle_path)
    if not path.exists():
        sys.exit(f"[ERROR] Bundle not found: {path}")
    with open(path, "rb") as f:
        return _PipelineUnpickler(f).load()


def predict_records(bundle, records: list[dict]) -> list[dict]:
    internal_keys = {"_expected"}
    clean = [{k: v for k, v in r.items() if k not in internal_keys} for r in records]
    df = pd.DataFrame(clean)

    labels = bundle.predict_labels(df)
    probas = bundle.predict_proba(df)

    results = []
    for i, rec in enumerate(records):
        prob_row = {
            bundle.label_names[j]: round(float(p), 4)
            for j, p in enumerate(probas[i])
        }
        results.append({
            "index": i,
            "prediction": labels[i],
            "probabilities": prob_row,
            "expected": rec.get("_expected"),
        })
    return results


def print_results(results: list[dict], model_name: str = "") -> None:
    sep = "-" * 56
    print("\n" + "=" * 56)
    print("  Credit Score Inference Results")
    if model_name:
        print(f"  Model: {model_name}")
    print("=" * 56)
    for r in results:
        pred = r["prediction"]
        probs = r["probabilities"]
        expected = r["expected"]
        status = ""
        if expected:
            status = "  [OK]" if pred == expected else f"  [FAIL] (expected {expected})"
        print(f"\n  Record #{r['index'] + 1}  ->  {pred}{status}")
        print(f"  {sep}")
        for label, p in probs.items():
            bar = "#" * int(p * 30)
            print(f"    {label:<10} {p:.4f}  {bar}")
    print("\n" + "=" * 56 + "\n")


def run_selftest(bundle) -> None:
    print(f"Running self-test with {len(SELFTEST_RECORDS)} records...")
    results = predict_records(bundle, SELFTEST_RECORDS)
    print_results(results, model_name=bundle.model_name)
    passed = sum(1 for r in results if r["prediction"] == r["expected"])
    total = len(results)
    print(f"Self-test: {passed}/{total} predictions matched expected class.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Credit Score inference -- loads best_model.pkl and predicts."
    )
    parser.add_argument(
        "--bundle",
        default=str(DEFAULT_BUNDLE),
        help="Path to the CreditScoreBundle pickle (default: artifacts/best_model.pkl)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run built-in self-test (one record per class).",
    )
    parser.add_argument(
        "--input",
        metavar="CSV",
        help="CSV file with raw records to score.",
    )
    parser.add_argument(
        "--json",
        dest="json_str",
        metavar="JSON",
        help="Inline JSON object or JSON array of raw records.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print results as JSON instead of pretty table.",
    )
    args = parser.parse_args()

    if not args.selftest and not args.input and not args.json_str:
        parser.print_help()
        sys.exit(0)

    bundle = load_bundle(args.bundle)
    print(f"[OK] Loaded bundle: model={bundle.model_name!r}, "
          f"features={len(bundle.feature_columns)}")

    if args.selftest:
        run_selftest(bundle)

    if args.input:
        df = pd.read_csv(args.input)
        records = df.to_dict(orient="records")
        results = predict_records(bundle, records)
        if args.output_json:
            print(json.dumps(results, indent=2))
        else:
            print_results(results, model_name=bundle.model_name)

    if args.json_str:
        try:
            parsed = json.loads(args.json_str)
        except json.JSONDecodeError as e:
            sys.exit(f"[ERROR] Invalid JSON: {e}")
        if isinstance(parsed, dict):
            parsed = [parsed]
        results = predict_records(bundle, parsed)
        if args.output_json:
            print(json.dumps(results, indent=2))
        else:
            print_results(results, model_name=bundle.model_name)


if __name__ == "__main__":
    main()
