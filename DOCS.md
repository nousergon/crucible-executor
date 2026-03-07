# Alpha Engine Executor — Full Setup Guide

End-to-end setup guide for an external developer standing up their own instance of the executor stack. Covers AWS infrastructure, EC2 configuration, IB Gateway, IBC, and the executor itself.

---

## Architecture overview

```
Research pipeline (separate repo)
        │
        ▼
S3: signals/{date}/signals.json          ← written by research pipeline
        │
        ▼ (read at market open)
EC2 (t3.small, Amazon Linux 2023)
  └── IB Gateway (paper, port 4002)
  └── executor/main.py  (cron 13:30 UTC weekdays)
        │  reads signals, places orders via ib_insync
        ▼
  trades.db (SQLite, local + S3 backup)
        │
  executor/eod_reconcile.py  (cron 21:05 UTC weekdays)
        │  NAV vs SPY, sends EOD email
        ▼
AWS SES → your inbox
```

---

## Prerequisites

- AWS account with programmatic access
- Interactive Brokers account (paper trading account sufficient)
- IB Gateway credentials (username + password)
- A verified email address in AWS SES
- SSH key pair created in EC2
- A signals pipeline that writes `signals/{date}/signals.json` to S3 — see [alpha-engine-research](https://github.com/cipher813/alpha-engine-research) for the reference implementation

---

## 1. AWS — S3 buckets

Create two S3 buckets in the same region (us-east-1 recommended):

| Bucket | Purpose |
|---|---|
| `your-research-bucket` | Receives `signals/{date}/signals.json` from research pipeline |
| `your-executor-bucket` | Receives `trades/trades_{date}.db` backups from executor |

Both buckets should block all public access (default). No special bucket policies needed — access is controlled via the EC2 IAM role (step 3).

```bash
aws s3 mb s3://your-research-bucket --region us-east-1
aws s3 mb s3://your-executor-bucket --region us-east-1
```

---

## 2. AWS — SES email

Verify the email address you want to send EOD reports from:

```bash
aws ses verify-email-identity --email-address you@example.com --region us-east-1
```

Check your inbox and click the verification link. Confirm it worked:

```bash
aws ses get-identity-verification-attributes --identities you@example.com --region us-east-1
```

`VerificationStatus` should be `Success`. If your AWS account is still in SES sandbox mode, you must also verify every recipient address the same way.

---

## 3. AWS — EC2 instance

### Launch

- **AMI:** Amazon Linux 2023 (x86_64)
- **Instance type:** t3.small (2 vCPU, 2 GB RAM — minimum for IB Gateway)
- **Region:** us-east-1 (or match your S3 buckets)
- **Storage:** 20 GB gp3
- **Key pair:** create or select an existing `.pem` key
- **Security group:** SSH (port 22) from your IP only — IB Gateway runs on localhost, no inbound needed

```bash
# Confirm the instance is running
aws ec2 describe-instances --region us-east-1 \
    --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]' \
    --output table
```

### IAM instance role

Create an IAM role and attach it to the EC2 instance so the executor can access S3 and SES without hardcoded credentials.

**Trust policy** (save as `trust-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ec2.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

**Permissions policy** (save as `executor-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::your-research-bucket",
        "arn:aws:s3:::your-research-bucket/signals/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::your-executor-bucket",
        "arn:aws:s3:::your-executor-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["ses:SendEmail", "ses:SendRawEmail"],
      "Resource": "*"
    }
  ]
}
```

```bash
# Create role
aws iam create-role --role-name alpha-engine-executor-role \
    --assume-role-policy-document file://trust-policy.json

# Attach permissions
aws iam put-role-policy --role-name alpha-engine-executor-role \
    --policy-name executor-policy \
    --policy-document file://executor-policy.json

# Create instance profile and attach role
aws iam create-instance-profile --instance-profile-name alpha-engine-executor-profile
aws iam add-role-to-instance-profile \
    --instance-profile-name alpha-engine-executor-profile \
    --role-name alpha-engine-executor-role

# Attach to your EC2 instance
aws ec2 associate-iam-instance-profile \
    --instance-id i-YOUR_INSTANCE_ID \
    --iam-instance-profile Name=alpha-engine-executor-profile \
    --region us-east-1
```

---

## 4. EC2 — base setup

SSH in and run initial setup:

```bash
ssh -i ~/.ssh/your-key.pem ec2-user@YOUR_EC2_IP
```

```bash
sudo yum update -y && sudo yum install -y git xorg-x11-server-Xvfb
```

---

## 5. EC2 — IB Gateway

IB Gateway requires a display (even on a headless server). Xvfb provides a virtual one.

### Download IB Gateway

Download the **standalone** Linux installer from the IBKR website — search for "IB Gateway standalone download". At time of writing, version 10.44 is tested and working.

```bash
chmod +x ibgateway-standalone-linux-x64.sh
./ibgateway-standalone-linux-x64.sh -q -dir ~/ibgateway
```

This installs into `~/ibgateway/` and places a bundled JRE under `~/.local/share/i4j_jres/`. **Always use this bundled JRE** — IB Gateway requires Azul Zulu 17 and will fail with system Java.

Find the bundled JRE path:
```bash
find ~/.local/share/i4j_jres -name java -type f
```

Create the version symlink directory IBC expects:
```bash
mkdir -p ~/ibgateway/1044
ln -s ~/ibgateway/jars ~/ibgateway/1044/jars
ln -s ~/ibgateway/.install4j ~/ibgateway/1044/.install4j
ln -s ~/ibgateway/ibgateway.vmoptions ~/ibgateway/1044/ibgateway.vmoptions
```

---

## 6. EC2 — IBC

IBC automates the IB Gateway login so it can run headlessly via systemd.

```bash
# Download IBC v3.23.0
curl -L https://github.com/IbcAlpha/IBC/releases/download/3.23.0/IBCLinux-3.23.0.zip -o ibc.zip
unzip ibc.zip -d ~/ibc
chmod +x ~/ibc/*.sh ~/ibc/scripts/*.sh
```

### Configure IBC

Edit `~/ibc/config.ini` — the key settings:

```ini
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper
AcceptNonBrokerageAccountWarning=yes
ReadOnlyApi=no
```

### Configure gatewaystart.sh

Edit `~/ibc/gatewaystart.sh` — set these variables at the top:

```bash
TWS_MAJOR_VRSN=1044
IBC_INI=~/ibc/config.ini
IBC_PATH=/home/ec2-user/ibc
TWS_PATH=/home/ec2-user
JAVA_PATH=          # leave blank — use bundled JRE via pref_jre.cfg
```

IBC discovers the bundled JRE via a `pref_jre.cfg` file written by the installer. Leaving `JAVA_PATH` blank lets IBC find it automatically. Do **not** point this at system Java.

---

## 7. EC2 — systemd services

Create two systemd services so IB Gateway starts automatically on boot.

### Xvfb (virtual display)

```bash
sudo tee /etc/systemd/system/xvfb.service << 'EOF'
[Unit]
Description=X Virtual Framebuffer
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :1 -screen 0 1024x768x16
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### IB Gateway

```bash
sudo tee /etc/systemd/system/ibgateway.service << 'EOF'
[Unit]
Description=IB Gateway via IBC
After=xvfb.service
Requires=xvfb.service

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/bin/bash /home/ec2-user/ibc/gatewaystart.sh -inline
Environment=DISPLAY=:1
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable xvfb ibgateway
sudo systemctl start xvfb && sudo systemctl start ibgateway
```

Verify (allow ~30s for IB Gateway to fully initialize):

```bash
sudo systemctl status ibgateway
```

IB Gateway logs are available at `~/ibc/logs/`.

---

## 8. EC2 — executor setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/alpha-engine.git ~/alpha-engine
cd ~/alpha-engine

# Create venv and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Create config from template
cp config/risk.yaml.example config/risk.yaml
```

Edit `config/risk.yaml` and fill in your values:

```yaml
signals_bucket: "your-research-bucket"
trades_bucket:  "your-executor-bucket"
email_sender:   "you@example.com"
email_recipients:
  - "you@example.com"
db_path: "/home/ec2-user/alpha-engine/trades.db"
```

All other parameters (position limits, drawdown thresholds, etc.) are pre-set to sensible defaults — review and adjust before going live.

Test the connection:

```bash
cd ~/alpha-engine && .venv/bin/python executor/connection_test.py
```

Expected output: `Connected: True` followed by account summary.

---

## 9. EC2 — cron jobs

```bash
crontab -e
```

Add (times are UTC):

```
30 13 * * 1-5  cd /home/ec2-user/alpha-engine && .venv/bin/python executor/main.py >> /var/log/executor.log 2>&1
5 21 * * 1-5   cd /home/ec2-user/alpha-engine && .venv/bin/python executor/eod_reconcile.py >> /var/log/eod.log 2>&1
```

| Cron time (UTC) | ET standard | ET daylight | Purpose |
|---|---|---|---|
| 13:30 UTC | 8:30am ET | 9:30am ET | Morning run at/near market open |
| 21:05 UTC | 4:05pm ET | 5:05pm ET | EOD reconciliation after close |

> Note: during ET standard time (Nov–Mar), the morning run fires at 8:30am ET — 60 minutes before market open. Signals will exist from the research pipeline but no prices will be available from IBKR until 9:30am. Adjust to `30 14 * * 1-5` (9:30am ET standard) if you want to run exactly at open year-round.

Create the log files:

```bash
sudo touch /var/log/executor.log /var/log/eod.log
sudo chown ec2-user:ec2-user /var/log/executor.log /var/log/eod.log
```

---

## 10. Signals interface

The executor reads `signals/{YYYY-MM-DD}/signals.json` from the research S3 bucket each morning. If today's file is missing it falls back up to 5 calendar days (skipping weekends) with a warning log.

[alpha-engine-research](https://github.com/cipher813/alpha-engine-research) is the reference implementation — a LangGraph pipeline that runs on AWS Lambda each trading day and writes a compliant signals file to S3. You can use it directly or build your own source that conforms to the schema below.

The file must conform to this schema:

```json
{
  "date": "YYYY-MM-DD",
  "market_regime": "bull | neutral | bear | caution",
  "sector_ratings": {
    "Technology": {
      "rating": "overweight | market_weight | underweight",
      "modifier": 1.2,
      "rationale": "..."
    }
  },
  "universe": [
    {
      "ticker": "PLTR",
      "sector": "Technology",
      "signal": "ENTER | EXIT | REDUCE | HOLD",
      "rating": "BUY | HOLD | SELL",
      "score": 78.5,
      "conviction": "rising | stable | declining",
      "price_target_upside": 0.18,
      "thesis_summary": "..."
    }
  ],
  "buy_candidates": []
}
```

`buy_candidates` can duplicate entries from `universe` with higher priority — the executor deduplicates by ticker, preferring `buy_candidates`.

---

## 11. Verification checklist

Run these in order after setup:

```bash
# 1. IB Gateway running
sudo systemctl status ibgateway

# 2. Port 4002 open
ss -tlnp | grep 4002

# 3. IBKR connection
cd ~/alpha-engine && .venv/bin/python executor/connection_test.py

# 4. S3 access — write a test signal and read it back
aws s3 cp /dev/stdin s3://your-research-bucket/signals/test/signals.json --content-type application/json <<< '{"date":"test","market_regime":"neutral","sector_ratings":{},"universe":[],"buy_candidates":[]}'
aws s3 ls s3://your-research-bucket/signals/test/

# 5. Full dry run
cd ~/alpha-engine && .venv/bin/python executor/main.py --dry-run
```

---

## Troubleshooting

**IB Gateway won't start**
Check `~/ibc/logs/` for the session log. Common causes: wrong `TWS_MAJOR_VRSN`, wrong JRE (must be bundled Zulu 17), display not available (Xvfb must be running first).

**`Connected: False` from connection_test.py**
IB Gateway takes ~30s to fully initialize after the service starts. Also check that `ReadOnlyApi=no` is set in `~/ibc/config.ini` — read-only mode blocks ib_insync from connecting.

**`NoSuchKey` on signals read**
The research pipeline hasn't written a file for the requested date. The executor will fall back to the most recent available date (up to 5 days). Check the research pipeline logs if this persists.

**SES send fails**
Confirm the sender address is verified (`aws ses get-identity-verification-attributes`). If the AWS account is in SES sandbox, recipient addresses must also be verified.

**Orders blocked by risk guard**
Check the log for the specific rule that fired (score, conviction, drawdown, position size, sector exposure). All thresholds are configurable in `config/risk.yaml`.
