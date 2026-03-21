#!/bin/bash
# Create EventBridge scheduled rules for:
#   1. Start trading instance weekdays 6:15 AM PT
#   2. Stop trading instance weekdays 1:30 PM PT
#   3. Launch backtester spot instance Mondays 08:00 UTC (via SSM on micro)
#
# Prerequisites:
#   - TRADING_INSTANCE_ID set (the t3.small trading instance)
#   - MICRO_INSTANCE_ID set (the t3.micro dashboard instance)
#   - IAM role for EventBridge to call EC2 + SSM (created below if missing)
#
# Usage:
#   TRADING_INSTANCE_ID=i-xxx MICRO_INSTANCE_ID=i-yyy bash infrastructure/setup-eventbridge.sh

set -euo pipefail

REGION="us-east-1"

if [ -z "${TRADING_INSTANCE_ID:-}" ] || [ -z "${MICRO_INSTANCE_ID:-}" ]; then
    echo "ERROR: Both TRADING_INSTANCE_ID and MICRO_INSTANCE_ID must be set"
    echo "Usage: TRADING_INSTANCE_ID=i-xxx MICRO_INSTANCE_ID=i-yyy bash $0"
    exit 1
fi

echo "=== EventBridge Scheduler Setup ==="
echo "Trading instance: ${TRADING_INSTANCE_ID}"
echo "Micro instance:   ${MICRO_INSTANCE_ID}"
echo "Region:           ${REGION}"
echo ""

# ── 1. IAM role for EventBridge ──────────────────────────────────────────────
ROLE_NAME="alpha-engine-eventbridge-role"

if ! aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating IAM role: ${ROLE_NAME}..."

    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "scheduler.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' \
        --region "$REGION"

    # EC2 start/stop permissions (scoped to trading instance)
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "ec2-trading-lifecycle" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": [\"ec2:StartInstances\", \"ec2:StopInstances\"],
                \"Resource\": \"arn:aws:ec2:${REGION}:*:instance/${TRADING_INSTANCE_ID}\"
            }]
        }" \
        --region "$REGION"

    # SSM send-command permissions (scoped to micro instance)
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "ssm-backtester-launch" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": \"ssm:SendCommand\",
                \"Resource\": [
                    \"arn:aws:ec2:${REGION}:*:instance/${MICRO_INSTANCE_ID}\",
                    \"arn:aws:ssm:${REGION}::document/AWS-RunShellScript\"
                ]
            }]
        }" \
        --region "$REGION"

    echo "IAM role created"
else
    echo "IAM role ${ROLE_NAME} already exists"
fi

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text --region "$REGION")
echo "Role ARN: ${ROLE_ARN}"

# ── 2. Start trading instance — weekdays 6:15 AM PT ─────────────────────────
# EventBridge Scheduler uses IANA timezone names
echo ""
echo "Creating schedule: start-trading-instance..."
aws scheduler create-schedule \
    --name "alpha-engine-start-trading" \
    --schedule-expression "cron(15 6 ? * MON-FRI *)" \
    --schedule-expression-timezone "America/Los_Angeles" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ec2:startInstances\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"InstanceIds\\\": [\\\"${TRADING_INSTANCE_ID}\\\"]}\"
    }" \
    --state ENABLED \
    --region "$REGION" \
    2>/dev/null || \
aws scheduler update-schedule \
    --name "alpha-engine-start-trading" \
    --schedule-expression "cron(15 6 ? * MON-FRI *)" \
    --schedule-expression-timezone "America/Los_Angeles" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ec2:startInstances\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"InstanceIds\\\": [\\\"${TRADING_INSTANCE_ID}\\\"]}\"
    }" \
    --state ENABLED \
    --region "$REGION"

echo "  Start: weekdays 6:15 AM PT"

# ── 3. Stop trading instance — weekdays 1:30 PM PT ──────────────────────────
echo "Creating schedule: stop-trading-instance..."
aws scheduler create-schedule \
    --name "alpha-engine-stop-trading" \
    --schedule-expression "cron(30 13 ? * MON-FRI *)" \
    --schedule-expression-timezone "America/Los_Angeles" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ec2:stopInstances\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"InstanceIds\\\": [\\\"${TRADING_INSTANCE_ID}\\\"]}\"
    }" \
    --state ENABLED \
    --region "$REGION" \
    2>/dev/null || \
aws scheduler update-schedule \
    --name "alpha-engine-stop-trading" \
    --schedule-expression "cron(30 13 ? * MON-FRI *)" \
    --schedule-expression-timezone "America/Los_Angeles" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ec2:stopInstances\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"InstanceIds\\\": [\\\"${TRADING_INSTANCE_ID}\\\"]}\"
    }" \
    --state ENABLED \
    --region "$REGION"

echo "  Stop: weekdays 1:30 PM PT"

# ── 4. Backtester spot launch — Mondays 08:00 UTC (via SSM on micro) ────────
echo "Creating schedule: backtester-spot-launch..."
BACKTEST_CMD="cd /home/ec2-user/alpha-engine-backtester && git pull --ff-only >> /var/log/backtester.log 2>&1 && . /home/ec2-user/.alpha-engine.env && bash infrastructure/spot_backtest.sh >> /var/log/backtester.log 2>&1"

aws scheduler create-schedule \
    --name "alpha-engine-backtester-weekly" \
    --schedule-expression "cron(0 8 ? * MON *)" \
    --schedule-expression-timezone "UTC" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ssm:sendCommand\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"DocumentName\\\": \\\"AWS-RunShellScript\\\", \\\"InstanceIds\\\": [\\\"${MICRO_INSTANCE_ID}\\\"], \\\"Parameters\\\": {\\\"commands\\\": [\\\"sudo -u ec2-user bash -c '${BACKTEST_CMD}'\\\"]} }\"
    }" \
    --state ENABLED \
    --region "$REGION" \
    2>/dev/null || \
aws scheduler update-schedule \
    --name "alpha-engine-backtester-weekly" \
    --schedule-expression "cron(0 8 ? * MON *)" \
    --schedule-expression-timezone "UTC" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{
        \"Arn\": \"arn:aws:scheduler:::aws-sdk:ssm:sendCommand\",
        \"RoleArn\": \"${ROLE_ARN}\",
        \"Input\": \"{\\\"DocumentName\\\": \\\"AWS-RunShellScript\\\", \\\"InstanceIds\\\": [\\\"${MICRO_INSTANCE_ID}\\\"], \\\"Parameters\\\": {\\\"commands\\\": [\\\"sudo -u ec2-user bash -c '${BACKTEST_CMD}'\\\"]} }\"
    }" \
    --state ENABLED \
    --region "$REGION"

echo "  Backtester: Mondays 08:00 UTC (SSM on micro)"

echo ""
echo "=== EventBridge Setup Complete ==="
echo ""
echo "Schedules created:"
echo "  alpha-engine-start-trading      weekdays 6:15 AM PT"
echo "  alpha-engine-stop-trading       weekdays 1:30 PM PT"
echo "  alpha-engine-backtester-weekly  Mondays 08:00 UTC"
echo ""
echo "Verify: aws scheduler list-schedules --region ${REGION}"
