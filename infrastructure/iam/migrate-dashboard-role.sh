#!/usr/bin/env bash
#
# migrate-dashboard-role.sh — One-shot migration that gives the dashboard EC2
# box its own least-privilege IAM role, splitting it off the shared
# `alpha-engine-executor-role` (which was the instance profile for BOTH the
# trading box and the dashboard box — a god-role spanning four projects).
#
# Target end state:
#   * NEW role  alpha-engine-dashboard-role  — 8 inline policies the dashboard
#     box actually uses (S3, SSM read, SFN read, SNS, CloudWatch, cyphering
#     SSM read, mnemon S3) + AmazonSSMManagedInstanceCore. Trust: ec2.
#   * dashboard box (i-09b539c844515d549) uses alpha-engine-dashboard-profile.
#   * trading role alpha-engine-executor-role loses the 3 dashboard-only
#     policies (cyphering-signal-ssm-read, alpha-engine-dashboard-sfn-read,
#     mnemon-s3-access) — the trading box provably does not use them.
#
# The role/policy SOURCE OF TRUTH is the codified JSON under
# infrastructure/iam/<role>/ — this script only bootstraps the new role +
# instance profile and repoints the box; ongoing policy edits go through
# apply.sh + check-drift.py like every other codified role.
#
# Run order (each step is idempotent; verify between swap and trim):
#   1. ./migrate-dashboard-role.sh create        # additive — create role+profile+policies, grant CI read
#   2. ./migrate-dashboard-role.sh swap          # repoint the live box (reversible via `rollback`)
#   3.   ... verify dashboard + cyphering site + box-health alerts ...
#   4. ./migrate-dashboard-role.sh trim-executor # remove dashboard-only policies from trading role
#
# Any step accepts --dry-run to print the planned AWS calls without executing.
# `rollback` repoints the box back to alpha-engine-executor-profile.
# `status`   prints the current association + both roles' inline policies.
#
# Prerequisites: AWS creds with iam:CreateRole, iam:PutRolePolicy,
# iam:AttachRolePolicy, iam:CreateInstanceProfile, iam:AddRoleToInstanceProfile,
# iam:DeleteRolePolicy, ec2:DescribeIamInstanceProfileAssociations,
# ec2:ReplaceIamInstanceProfileAssociation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"

DASHBOARD_ROLE="alpha-engine-dashboard-role"
DASHBOARD_PROFILE="alpha-engine-dashboard-profile"
EXECUTOR_PROFILE="alpha-engine-executor-profile"
EXECUTOR_ROLE="alpha-engine-executor-role"
CI_ROLE="github-actions-iam-drift-check"
DASHBOARD_INSTANCE="${DASHBOARD_INSTANCE:-i-09b539c844515d549}"

# Dashboard-only inline policies to remove from the trading role in trim step.
TRIM_POLICIES=(cyphering-signal-ssm-read alpha-engine-dashboard-sfn-read mnemon-s3-access)

DRY_RUN=0

run() {
  # Echo every AWS call; execute unless --dry-run.
  echo "  + $*"
  if [ "$DRY_RUN" = 0 ]; then
    "$@"
  fi
}

cmd_create() {
  echo "[create] role + profile + policies (idempotent, additive)"

  if aws iam get-role --role-name "$DASHBOARD_ROLE" >/dev/null 2>&1; then
    echo "  role $DASHBOARD_ROLE already exists — skipping create-role"
  else
    run aws iam create-role --role-name "$DASHBOARD_ROLE" \
      --assume-role-policy-document "file://$SCRIPT_DIR/$DASHBOARD_ROLE/trust-policy.json" \
      --description "Least-privilege role for the dashboard/monitoring EC2 box (split from executor-role)" \
      --region "$REGION"
  fi

  echo "[create] applying codified inline + trust + managed for $DASHBOARD_ROLE"
  local dry_flag=""
  [ "$DRY_RUN" = 1 ] && dry_flag="--dry-run"
  bash "$SCRIPT_DIR/apply.sh" --role "$DASHBOARD_ROLE" $dry_flag

  if aws iam get-instance-profile --instance-profile-name "$DASHBOARD_PROFILE" >/dev/null 2>&1; then
    echo "  instance profile $DASHBOARD_PROFILE already exists — skipping"
  else
    run aws iam create-instance-profile --instance-profile-name "$DASHBOARD_PROFILE" --region "$REGION"
  fi

  # add-role-to-instance-profile is not idempotent — guard it.
  if aws iam get-instance-profile --instance-profile-name "$DASHBOARD_PROFILE" \
       --query "InstanceProfile.Roles[].RoleName" --output text 2>/dev/null | grep -qw "$DASHBOARD_ROLE"; then
    echo "  role already attached to $DASHBOARD_PROFILE — skipping"
  else
    run aws iam add-role-to-instance-profile --instance-profile-name "$DASHBOARD_PROFILE" \
      --role-name "$DASHBOARD_ROLE" --region "$REGION"
  fi

  echo "[create] updating CI drift-check read role (new actions + dashboard role ARN)"
  bash "$SCRIPT_DIR/apply.sh" --role "$CI_ROLE" --policy iam-readonly $dry_flag

  echo "[create] done. Next: run 'swap' to repoint $DASHBOARD_INSTANCE."
}

_current_association() {
  aws ec2 describe-iam-instance-profile-associations \
    --filters "Name=instance-id,Values=$DASHBOARD_INSTANCE" \
    --query "IamInstanceProfileAssociations[?State=='associated'].AssociationId" \
    --output text --region "$REGION"
}

cmd_swap() {
  echo "[swap] repointing $DASHBOARD_INSTANCE -> $DASHBOARD_PROFILE"
  local assoc
  assoc="$(_current_association)"
  if [ -z "$assoc" ]; then
    echo "ERROR: no associated instance profile found for $DASHBOARD_INSTANCE" >&2
    exit 1
  fi
  echo "  current association: $assoc"
  run aws ec2 replace-iam-instance-profile-association \
    --association-id "$assoc" \
    --iam-instance-profile "Name=$DASHBOARD_PROFILE" \
    --region "$REGION"
  echo "[swap] done. Role creds refresh on the box within minutes (or reboot to force)."
  echo "       VERIFY dashboard + cyphering site + box-health alerts before 'trim-executor'."
}

cmd_rollback() {
  echo "[rollback] repointing $DASHBOARD_INSTANCE back to $EXECUTOR_PROFILE"
  local assoc
  assoc="$(_current_association)"
  if [ -z "$assoc" ]; then
    echo "ERROR: no associated instance profile found for $DASHBOARD_INSTANCE" >&2
    exit 1
  fi
  run aws ec2 replace-iam-instance-profile-association \
    --association-id "$assoc" \
    --iam-instance-profile "Name=$EXECUTOR_PROFILE" \
    --region "$REGION"
  echo "[rollback] done."
}

cmd_trim_executor() {
  echo "[trim-executor] removing dashboard-only inline policies from $EXECUTOR_ROLE"
  echo "  (reversible: re-run apply.sh from the codified JSON — but these files now live under $DASHBOARD_ROLE/)"
  local p
  for p in "${TRIM_POLICIES[@]}"; do
    if aws iam get-role-policy --role-name "$EXECUTOR_ROLE" --policy-name "$p" >/dev/null 2>&1; then
      run aws iam delete-role-policy --role-name "$EXECUTOR_ROLE" --policy-name "$p" --region "$REGION"
    else
      echo "  $p not present on $EXECUTOR_ROLE — already trimmed"
    fi
  done
  echo "[trim-executor] done. Run check-drift.py to confirm both roles are clean."
}

cmd_status() {
  echo "=== instance profile association for $DASHBOARD_INSTANCE ==="
  aws ec2 describe-iam-instance-profile-associations \
    --filters "Name=instance-id,Values=$DASHBOARD_INSTANCE" \
    --query "IamInstanceProfileAssociations[].{Assoc:AssociationId,Profile:IamInstanceProfile.Arn,State:State}" \
    --output table --region "$REGION"
  echo "=== $DASHBOARD_ROLE inline policies ==="
  aws iam list-role-policies --role-name "$DASHBOARD_ROLE" --query PolicyNames --output json 2>/dev/null || echo "  (role does not exist yet)"
  echo "=== $EXECUTOR_ROLE inline policies ==="
  aws iam list-role-policies --role-name "$EXECUTOR_ROLE" --query PolicyNames --output json
}

SUBCMD="${1:-}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$SUBCMD" in
  create)        cmd_create ;;
  swap)          cmd_swap ;;
  rollback)      cmd_rollback ;;
  trim-executor) cmd_trim_executor ;;
  status)        cmd_status ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
