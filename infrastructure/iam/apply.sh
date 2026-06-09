#!/usr/bin/env bash
#
# apply.sh — Apply codified IAM (inline policies + trust policy + managed
# attachments) in this directory tree to their matching roles.
#
# Layout (directory-per-role to support multiple inline policies per role):
#
#   infrastructure/iam/<role-name>/<policy-name>.json
#
# Each JSON file is an inline policy document. The directory name is the IAM
# role name. The filename (minus .json) is the inline policy name. This keeps
# the 1:1 file→inline-policy mapping that alpha-engine-data established, while
# accommodating roles that already have multiple inline policies in prod
# (the executor role has 9 as of 2026-04-27).
#
# This is intentionally low-ceremony — no CloudFormation, no Terraform.
# Role CREATION is NOT managed here (the script only updates roles that
# already exist). Two reserved per-role files extend coverage beyond inline
# policies (mirrors check-drift.py):
#
#   <role>/trust-policy.json      — applied via update-assume-role-policy
#   <role>/managed-policies.json  — JSON array of ARNs; applied via
#                                   attach-role-policy. Managed policies
#                                   attached on AWS but NOT in the array are
#                                   NOT auto-detached — apply.sh WARNs and
#                                   leaves the detach to a human (destructive).
#
# Usage:
#   ./infrastructure/iam/apply.sh                                # apply every role
#   ./infrastructure/iam/apply.sh --role alpha-engine-executor-role
#                                                                # one role (inline+trust+managed)
#   ./infrastructure/iam/apply.sh --role <role> --policy <name>  # one specific inline policy
#   ./infrastructure/iam/apply.sh --dry-run                      # print planned commands
#
# Prerequisites:
#   - AWS CLI configured with iam:PutRolePolicy, iam:UpdateAssumeRolePolicy,
#     iam:AttachRolePolicy, iam:ListAttachedRolePolicies on the target roles
#   - The target IAM roles already exist in AWS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"

TRUST_FILE="trust-policy.json"
MANAGED_FILE="managed-policies.json"

DRY_RUN=0
TARGET_ROLE=""
TARGET_POLICY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --role)
      TARGET_ROLE="$2"
      shift 2
      ;;
    --policy)
      TARGET_POLICY="$2"
      shift 2
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# Apply one inline policy file (NOT trust / managed — those are reserved).
apply_inline() {
  local file="$1"
  local role
  role="$(basename "$(dirname "$file")")"
  local policy_name
  policy_name="$(basename "$file" .json)"

  if ! python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
    echo "ERROR: $file is not valid JSON — skipping" >&2
    return 1
  fi

  echo "Applying inline $file -> role=$role policy=$policy_name"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] aws iam put-role-policy --role-name $role --policy-name $policy_name --policy-document file://$file --region $REGION"
    return 0
  fi

  aws iam put-role-policy \
    --role-name "$role" \
    --policy-name "$policy_name" \
    --policy-document "file://$file" \
    --region "$REGION"
  echo "  OK"
}

# Apply the role's trust policy (opt-in; no-op if trust-policy.json absent).
apply_trust() {
  local role="$1"
  local file="$role/$TRUST_FILE"
  [[ -f "$file" ]] || return 0

  if ! python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
    echo "ERROR: $file is not valid JSON — skipping" >&2
    return 1
  fi

  echo "Applying trust $file -> role=$role"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] aws iam update-assume-role-policy --role-name $role --policy-document file://$file --region $REGION"
    return 0
  fi

  aws iam update-assume-role-policy \
    --role-name "$role" \
    --policy-document "file://$file" \
    --region "$REGION"
  echo "  OK"
}

# Attach the role's managed policies (opt-in; no-op if managed-policies.json
# absent). Additive only — extras attached on AWS but not codified are WARNed,
# never auto-detached (detach is destructive and left to a human).
apply_managed() {
  local role="$1"
  local file="$role/$MANAGED_FILE"
  [[ -f "$file" ]] || return 0

  local source_arns
  if ! source_arns="$(python3 -c "import json,sys; d=json.load(open('$file')); assert isinstance(d,list); print('\n'.join(d))" 2>/dev/null)"; then
    echo "ERROR: $file is not a JSON array of ARNs — skipping" >&2
    return 1
  fi

  local live_arns
  # Tolerate the role not existing yet (e.g. --dry-run before create-role):
  # treat as no current attachments rather than hard-failing under pipefail.
  live_arns="$(aws iam list-attached-role-policies --role-name "$role" --region "$REGION" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null | tr '\t' '\n' || true)"

  # Attach any codified ARN not currently attached.
  local arn
  while IFS= read -r arn; do
    [[ -z "$arn" ]] && continue
    if grep -qxF "$arn" <<<"$live_arns"; then
      continue
    fi
    echo "Attaching managed $arn -> role=$role"
    if [ "$DRY_RUN" = 1 ]; then
      echo "  [dry-run] aws iam attach-role-policy --role-name $role --policy-arn $arn --region $REGION"
    else
      aws iam attach-role-policy --role-name "$role" --policy-arn "$arn" --region "$REGION"
      echo "  OK"
    fi
  done <<<"$source_arns"

  # Warn (do not detach) on managed policies attached but not codified.
  while IFS= read -r arn; do
    [[ -z "$arn" ]] && continue
    if ! grep -qxF "$arn" <<<"$source_arns"; then
      echo "  WARN: $role has managed policy attached but not codified: $arn (detach manually if intended)" >&2
    fi
  done <<<"$live_arns"
}

# Apply everything for one role: inline policies + trust + managed.
apply_role() {
  local role="$1"
  if [[ ! -d "$role" ]]; then
    echo "ERROR: role directory $role not found" >&2
    return 1
  fi
  local file
  for file in "$role"/*.json; do
    case "$(basename "$file")" in
      "$TRUST_FILE"|"$MANAGED_FILE") continue ;;  # reserved — handled below
    esac
    apply_inline "$file"
  done
  apply_trust "$role"
  apply_managed "$role"
}

cd "$SCRIPT_DIR"

shopt -s nullglob

if [[ -n "$TARGET_ROLE" && -n "$TARGET_POLICY" ]]; then
  case "$TARGET_POLICY" in
    trust-policy)   apply_trust "$TARGET_ROLE" ;;
    managed-policies) apply_managed "$TARGET_ROLE" ;;
    *)
      file="${TARGET_ROLE}/${TARGET_POLICY}.json"
      if [[ ! -f "$file" ]]; then
        echo "ERROR: $file not found" >&2
        exit 1
      fi
      apply_inline "$file"
      ;;
  esac
elif [[ -n "$TARGET_ROLE" ]]; then
  apply_role "$TARGET_ROLE"
else
  roles=( */ )
  if [[ ${#roles[@]} -eq 0 ]]; then
    echo "No role directories found under $SCRIPT_DIR"
    exit 0
  fi
  for role in "${roles[@]}"; do
    apply_role "${role%/}"
  done
fi
