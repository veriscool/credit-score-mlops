"""Deploy the trained CreditScoreBundle to a SageMaker real-time endpoint."""

import json

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


# ---- EDIT THESE ---------------------------------------------------------
BUCKET = "credit-score-yourname-1234"
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"
# -------------------------------------------------------------------------

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"  # matches scikit-learn>=1.2 pinned in requirements.txt

# One raw record per class, same shape the local Streamlit app / predict.py
# --selftest send. Used only for the post-deploy smoke test below.
SAMPLE_RECORD = {
    "Month": "September",
    "Age": 45,
    "Occupation": "Engineer",
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
}


def get_lab_role_arn() -> str:
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint (5-8 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    sample = {"instances": [SAMPLE_RECORD]}

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(sample),
    )
    print("\nSmoke test response (expect label 'Good'):")
    print(response["Body"].read().decode("utf-8"))

    print(
        f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.\n"
        f"Delete it before lab teardown: predictor.delete_endpoint()"
    )


if __name__ == "__main__":
    main()
