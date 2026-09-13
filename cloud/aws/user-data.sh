#!/bin/bash
# Bootstrap script for an EC2 instance hosting the Streamlit app.
# Paste into EC2 launch wizard -> Advanced details -> User data.
# Edit the variables below before launching. Requires an IAM instance
# profile with S3 read + sagemaker:InvokeEndpoint (LabInstanceProfile
# in AWS Academy Learner Lab already has both).

set -eu

# -------- EDIT THESE -----------------------------------------------------
APP_S3_URI="s3://<your-bucket>/deploy/streamlit_app.py"
APP_FILE="streamlit_app.py"
ENDPOINT_NAME="credit-score-endpoint"
# -------------------------------------------------------------------------

REGION="us-east-1"
APP_DIR="/opt/credit-app"
VENV_DIR="/opt/streamlit-venv"

dnf update -y
dnf install -y python3 python3-pip awscli

mkdir -p "$APP_DIR"
aws s3 cp "$APP_S3_URI" "$APP_DIR/$APP_FILE" --region "$REGION"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_DIR/$APP_FILE" ]; then
  echo "FATAL: $APP_DIR/$APP_FILE not found after S3 download."
  exit 1
fi

# Use a venv to avoid conflicts with rpm-managed system Python packages.
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install streamlit boto3

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_DIR
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service

sleep 5
if systemctl is-active --quiet streamlit; then
  touch "$APP_DIR/.userdata-success"
  chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
  echo "FATAL: streamlit service failed to start."
  journalctl -u streamlit -n 30 --no-pager || true
  exit 1
fi
