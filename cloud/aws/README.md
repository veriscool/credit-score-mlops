# Credit Score - AWS SageMaker + Streamlit Deployment

Deploys the same 3-class credit-score classifier (`Poor` / `Standard` / `Good`)
trained by [`../pipeline.py`](../pipeline.py) to AWS: package the
`CreditScoreBundle` (fitted preprocessor + trained model), deploy it as a
SageMaker real-time endpoint, and call that endpoint from a Streamlit app.

## Architecture

```
[Browser]
   |
   v
[Streamlit app]               <- runs on your laptop OR on EC2
   |
   |  boto3.invoke_endpoint()
   v
[SageMaker endpoint]          <- runs on AWS
   |
   v
[S3: model.tar.gz]
```

The SageMaker endpoint itself is not publicly reachable; every call requires
an AWS-signed request (SigV4), using either local AWS credentials or the EC2
instance's IAM instance profile. That's normal and expected. Public access is
provided by the **Streamlit app**, not the endpoint (Option B below: EC2 with
a public IP on port 8501, open to `0.0.0.0/0`).

## Repo layout

```
cloud/aws/
├── README.md
├── pipeline.py                # entry point: train all candidate models, compare, package the winner
├── deploy_endpoint.py         # creates the SageMaker endpoint
├── streamlit_app.py           # the UI (23-field credit form + presets, same as ../app/streamlit_app.py)
├── user-data.sh               # EC2 bootstrap script
├── model_artifact/            # output of pipeline.py: model.tar.gz (generated, not committed)
└── src/
    ├── data.py                # CreditDataPreprocessor + DataSplitter (leakage-safe, by Customer_ID)
    ├── models.py               # BaseModel + 6 candidate models (Optuna search space, early stopping)
    ├── evaluate.py             # ModelEvaluator, comparison table, winner selection
    ├── bundle.py               # CreditScoreBundle: fitted preprocessor + trained model in one object
    ├── inference.py            # SageMaker container entry point
    └── requirements.txt        # extra packages for the SageMaker container
```

This mirrors `../pipeline.py`'s four concerns (preprocessing, model zoo,
evaluation, and orchestration) split into separate files instead of one
module, which is what SageMaker's `source_dir="src"` convention expects.
`bundle.py` bundles the fitted preprocessor with whichever model wins, so a
raw record can be scored the same way at inference time as it was in
training, the same idea as the local deployment's `artifacts/best_model.pkl`.

### What each file does

`pipeline.py` is the orchestrator: it loads data, fits
`CreditDataPreprocessor`, splits with `DataSplitter`, builds every model from
`build_models()`, tunes, fits, and evaluates each, prints a comparison table,
picks the winner by macro-F1, wraps it in a `CreditScoreBundle`, and packages
`model_artifact/model.tar.gz`. Unlike `../pipeline.py` it has no MLflow
tracking or persistent Optuna storage; those live in the local training
pipeline, since this script's job is just the cloud build.

`src/data.py`, `src/models.py`, and `src/evaluate.py` are extracted from
`../pipeline.py`'s `CreditDataPreprocessor`/`DataSplitter`, `BaseModel` plus
the 6 model subclasses, and `ModelEvaluator` respectively: same
preprocessing, Optuna search spaces, SMOTE handling, and early-stopping logic
as the local pipeline, just split into files by concern. If you change
`../pipeline.py`'s logic, port the change here too, since there's no
automatic sync between the two (see [Future work](../README.md#future-work)
for the plan to fix that).

`src/bundle.py` holds `CreditScoreBundle`, which `pipeline.py` imports rather
than defining inline. That's what lets `src/inference.py` load the pickle
with a plain `pickle.load()` and no custom unpickler needed. Contrast this
with the local deployment's `../scripts/predict.py` /
`../app/streamlit_app.py`, which need a `_PipelineUnpickler` to remap
`__main__.CreditScoreBundle` because `../pipeline.py` runs as a script and
defines its classes inline (so they pickle under the `__main__` module).
Here, because `pipeline.py` imports `CreditScoreBundle` from `bundle.py`, the
pickle references the real module name `bundle` directly, which resolves
without special-casing under SageMaker's `source_dir="src"`.

`src/inference.py` is the SageMaker container entry point. It loads
`best_model.pkl`, forces the model onto CPU (in case the champion was trained
on GPU), and exposes the `model_fn`/`input_fn`/`predict_fn`/`output_fn`
contract SageMaker's SKLearn container expects. It parses raw 23-field
records (dicts, not flat float arrays), because `CreditDataPreprocessor`,
bundled inside the pickle, expects the same raw CSV columns the local app
sends. The container does the same preprocess-then-predict as
`CreditScoreBundle.predict_proba()` locally.

`streamlit_app.py` reuses the same form, presets, and styling as
`../app/streamlit_app.py`, with the local in-process prediction call replaced
by `boto3.client("sagemaker-runtime").invoke_endpoint()`.

## If you're using AWS Academy Learner Lab

A few constraints if you're deploying this from a Learner Lab sandbox rather
than a normal AWS account:

- Region is fixed to `us-east-1`.
- You use the pre-existing `LabRole` IAM role, and can't create custom roles.
- SageMaker endpoint instances are limited to medium/large/xlarge sizes (this
  project uses `ml.m5.large`).
- Credentials are temporary and need refreshing when the lab session ends.
- Endpoints bill per second, so delete them before ending the session.

---

## Part 1: Train and package the model

### 1.1 Open a SageMaker Notebook Instance

AWS Console → SageMaker → Notebook → Notebook instances → Create notebook
instance.

- Name: anything, e.g. `credit-score-notebook`
- Instance type: `ml.t3.medium`
- IAM role: whatever role your account uses for SageMaker (`LabRole` on
  Academy)

Wait for `InService`, then `Open JupyterLab`.

### 1.2 Upload the project files

Drag into the JupyterLab file browser:

- `pipeline.py`
- `deploy_endpoint.py`
- the `src/` folder (`data.py`, `models.py`, `evaluate.py`, `bundle.py`,
  `inference.py`, `requirements.txt`)
- the dataset CSV, named `data_A.csv` at the notebook's root, next to
  `pipeline.py` (see [`../data/README.md`](../data/README.md) for where to
  get it)

Open a terminal (File → New → Terminal).

### 1.3 Run the pipeline

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn optuna
python pipeline.py --data data_A.csv --n-trials 0
```

This trains all six candidate models (`xgboost`, `random_forest`, `lightgbm`,
`knn`, `extra_trees`, `decision_tree`) with their tuned default parameters (no
Optuna search, `--n-trials 0`), prints a comparison table, and packages the
macro-F1 winner:

```
Model              Macro F1   Accuracy
----------------------------------------
lightgbm             0.6745     0.6946
xgboost               ...        ...
...

Winner: lightgbm  (macro-F1=0.6745)
Packaged: model_artifact/model.tar.gz

Next steps:
  aws s3 mb s3://your-bucket-name --region us-east-1
  aws s3 cp model_artifact/model.tar.gz s3://your-bucket-name/credit-score/model.tar.gz
```

A full `--n-trials 0` run over all 25k rows takes well under a minute
end-to-end, so it comfortably fits a `ml.t3.medium` notebook instance too. To
train (and deploy) just one model instead of comparing all six, pass
`--models lightgbm`. To reproduce the local Optuna search here instead of
reusing tuned defaults, raise `--n-trials` (30 to match the local run); expect
this to take much longer than the `t3.medium`'s 2 vCPUs are comfortable with,
so consider a bigger instance type if you do this.

---

## Part 2: Create an S3 bucket and upload the model

```bash
BUCKET=credit-score-<your-name>-<4-random-digits>
aws s3 mb s3://$BUCKET --region us-east-1
aws s3 cp model_artifact/model.tar.gz s3://$BUCKET/credit-score/model.tar.gz
aws s3 ls s3://$BUCKET/credit-score/
```

Bucket names must be globally unique. The random suffix is not optional.

---

## Part 3: Deploy the SageMaker endpoint

### 3.1 Edit deploy_endpoint.py

```python
BUCKET = "credit-score-yourname-1234"   # the bucket you just created
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"
```

### 3.2 Run it

```bash
python deploy_endpoint.py
```

Endpoint creation takes 5 to 8 minutes. While waiting, `src/inference.py` is
worth reading; the four SageMaker contract functions are the main concept.

On success you'll see a smoke-test response for the built-in "Good" sample
record:

```json
{"probabilities": [[0.0745, 0.3747, 0.5508]], "predictions": [2], "labels": ["Good"]}
```

Verify in the console: SageMaker → Inference → Endpoints.

### 3.3 About the model file format

The bundle is a pickle (via `CreditScoreBundle.save()`). It's Python-only,
and it can execute arbitrary code on load, so never load one from an
untrusted source. The container needs every library that was in scope when
the bundle was created (see `src/requirements.txt`); LightGBM is the current
champion but XGBoost is listed too in case a future `pipeline.py --models
xgboost` run changes the champion. The unpickler needs the winning model's
library importable even though it doesn't touch the others at inference time.

---

## Part 4: Run the Streamlit app

Both options call the same SageMaker endpoint and produce identical results.

### Option A: Run Streamlit on your laptop

```bash
pip install streamlit boto3
```

Configure `~/.aws/credentials` and `~/.aws/config` with credentials that can
call `sagemaker-runtime:InvokeEndpoint`, then:

```bash
aws sts get-caller-identity   # confirm credentials work first
```

```powershell
$env:ENDPOINT_NAME = "credit-score-endpoint"
$env:AWS_REGION = "us-east-1"
streamlit run streamlit_app.py
```

Open the printed URL, click one of the Poor / Standard / Good preset buttons
in the sidebar (or fill the form yourself), then Predict Credit Score.

### Option B: Run Streamlit on EC2 (this is what makes the URL public)

#### B.1 Get `streamlit_app.py` onto the instance

`user-data.sh` downloads a single file from an S3 URI at boot. Set
`APP_S3_URI` in `user-data.sh` to wherever you've uploaded `streamlit_app.py`
(e.g. `s3://<your-bucket>/deploy/streamlit_app.py`), and grant the EC2
instance profile S3 read access to that path.

#### B.2 Launch the EC2 instance

EC2 → Launch instances.

- AMI: Amazon Linux 2023, Instance type: `t3.micro`
- Key pair: Proceed without a key pair (use EC2 Instance Connect), or attach
  one you control if you need SSH access
- Network settings → Edit:
  - SSH from: My IP
  - Custom TCP, port `8501`, source `0.0.0.0/0` (this is the setting that
    makes the app publicly accessible)
- Advanced details → IAM instance profile: an instance profile with S3 read +
  `sagemaker:InvokeEndpoint` (`LabInstanceProfile` on Academy); User data:
  paste `user-data.sh`

#### B.3 Wait and verify

```bash
curl -v http://<public-ip>:8501
```

Open `http://<public-ip>:8501` in a browser. This URL stays reachable as long
as the EC2 instance keeps running and its public IP doesn't change. Allocate
an Elastic IP if you need it to survive a stop/start.

#### B.4 Useful EC2 commands

```bash
ls /opt/credit-app/.userdata-success
sudo systemctl status streamlit
sudo journalctl -u streamlit -f
sudo systemctl restart streamlit
```

---

## Part 5: Teardown

Endpoints and EC2 instances bill while running, so tear them down when done:

```python
import boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
sm.delete_endpoint(EndpointName="credit-score-endpoint")
sm.delete_endpoint_config(EndpointConfigName="credit-score-endpoint")
```

Then: terminate the EC2 instance, stop/delete the notebook instance, and
empty and delete the S3 bucket:

```bash
aws s3 rm s3://$BUCKET --recursive
aws s3 rb s3://$BUCKET
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `aws sts get-caller-identity` fails | Credentials expired or not configured; re-check `~/.aws/credentials` |
| `AccessDenied` on `iam:PassRole` | The role you're deploying with can't pass the execution role to SageMaker; use the account's designated role (`LabRole` on Academy) |
| Endpoint stuck in `Creating` for a long time | Check CloudWatch logs (`/aws/sagemaker/Endpoints/<endpoint-name>`) for a container crash on boot |
| `ModuleNotFoundError` in CloudWatch logs | A package used by `src/inference.py` isn't in `src/requirements.txt` |
| `ImportError: numpy.core.multiarray failed to import` in CloudWatch | `src/requirements.txt` asked pip to upgrade pandas/numpy, which it shouldn't; the container's baked-in versions are binary-tested together (see the main README's [Challenges & fixes](../README.md#challenges--fixes)) |
| EC2 connection refused on port 8501 | Security group doesn't actually allow `0.0.0.0/0:8501` inbound, or the streamlit service isn't running (`sudo systemctl status streamlit`) |
| `NoCredentialsError` when running Streamlit locally | `~/.aws/credentials` isn't configured, or the session token expired |
| `ResourceNotFound` on `invoke_endpoint` | `ENDPOINT_NAME` doesn't match what you deployed, or the endpoint was deleted |
| `KeyError` / `AttributeError` inside `predict_fn` in CloudWatch logs | A record is missing a required raw field, or `Credit_History_Age` wasn't sent as the `"X Years and Y Months"` string; check the payload matches the schema below |

---

## Sample request

The endpoint accepts JSON only: one or more raw credit records (the same 23
fields the local Streamlit form and `scripts/predict.py --selftest` use), not
a flat numeric vector.

```json
{
  "instances": [
    {
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
      "Monthly_Balance": 2500.0
    }
  ]
}
```

Response (exact probabilities will vary run to run):

```json
{
  "probabilities": [[0.0745, 0.3747, 0.5508]],
  "predictions": [2],
  "labels": ["Good"]
}
```

`predictions` / `probabilities` are always ordered `[Poor, Standard, Good]`
(see `../pipeline.py`'s `LABEL_NAMES`).

---

## Going further

- **Add a model.** Add a new `BaseModel` subclass to `src/models.py` and list
  it in `build_models()`. Nothing else needs to change: `pipeline.py`'s
  comparison loop and `src/inference.py` are both generic over whichever
  bundle wins.
- **Reproduce the full local Optuna search in the cloud.** Raise `--n-trials`
  and pick a bigger notebook instance type.
- **Add authentication to the Streamlit app.** Right now anyone with the EC2
  public IP can call the model. Put it behind nginx with HTTP basic auth, or
  an ALB with Cognito.
- **Unify this with the local pipeline.** `src/data.py`, `src/models.py`, and
  `src/evaluate.py` are manually kept in sync with `../pipeline.py` today.
  See the main README's [Future work](../README.md#future-work).
