# IAM policies (alpha-engine executor)

Source-of-truth for the inline IAM policies on the executor's IAM roles.
Mirrors the `alpha-engine-data` codification pattern (one JSON file per
inline policy + an `apply.sh` runner) with a directory-per-role layout
since the executor role has multiple inline policies.

## Layout

```
infrastructure/iam/
├── apply.sh
├── README.md
└── <role-name>/
    ├── <policy-1>.json
    ├── <policy-2>.json
    └── ...
```

The directory name is the IAM role name; each JSON filename (minus `.json`)
is the inline policy name on that role.

## Roles managed here

- **`alpha-engine-executor-role`** — assumed by the trading EC2 instance
  (`ae-trading`) and any executor processes assuming it. 8 inline policies
  as of 2026-04-27 (was 9 — `alpha-engine-ssm-access` consolidated into
  `alpha-engine-ssm-read`, which already had the superset of actions).
  Trust policy + role creation are NOT managed here (out of scope for the
  flat-file approach).
- **`alpha-engine-step-functions-role`** — assumed by all three Step
  Functions (`alpha-engine-saturday-pipeline`, `alpha-engine-weekday-pipeline`,
  `alpha-engine-eod-pipeline`). One consolidated inline policy granting
  Lambda invoke, SSM run, EC2 start/stop on the trading instance, SNS
  publish, and CloudWatch Logs delivery. Codified 2026-05-04 after an
  asymmetric `ec2:StartInstances` / `ec2:StopInstances` grant let the EOD
  SF stall on `StopTradingInstance` for an entire afternoon.
- **`alpha-engine-eventbridge-sfn-role`** — assumed by EventBridge to
  start the saturday + weekday SFN executions on cron. One inline policy
  granting `states:StartExecution` on both state machine ARNs. Codified
  2026-05-06 after a third recurrence of the same regression: the
  `alpha-engine-data` deploy scripts each contained an inline
  `aws iam put-role-policy` against this role, but only the weekday
  script listed both ARNs — the saturday script overwrote the policy
  with the saturday ARN alone, dropping weekday's grant. Inline blocks
  are removed from those scripts; `apply.sh` is the only writer now.
- **`github-actions-iam-drift-check`** — assumed by GitHub Actions via
  OIDC for the daily IAM-drift-check workflow. Single inline policy
  granting `iam:ListRolePolicies` + `iam:GetRolePolicy` scoped to the
  roles this directory manages. Trust policy: `repo:cipher813/alpha-engine`
  (main + pull_request).

## Out of scope (not codified here)

- Trust policies (`AssumeRolePolicyDocument`) — those are role creation,
  managed manually
- Managed policies (e.g. `AmazonSSMManagedInstanceCore` is attached to
  `alpha-engine-executor-role`) — managed manually via attach commands
- Role creation itself — pre-existing, managed manually

## Usage

```bash
# Apply every policy in this directory tree
./infrastructure/iam/apply.sh

# Apply every policy on one role
./infrastructure/iam/apply.sh --role alpha-engine-executor-role

# Apply one specific policy
./infrastructure/iam/apply.sh --role alpha-engine-executor-role --policy alpha-engine-cloudwatch-metrics

# Print planned commands without executing
./infrastructure/iam/apply.sh --dry-run
```

`apply.sh` calls `aws iam put-role-policy`, which is idempotent — re-running
overwrites the existing inline policy on the role. To remove a policy you
codified here, delete the file AND run `aws iam delete-role-policy` manually
(removal is not yet automated to avoid an `apply.sh` invocation accidentally
wiping policies whose JSON file was deleted in a stale checkout).

## Drift detection

`check-drift.py` diffs the codified state against AWS for every role
directory under `infrastructure/iam/`. It checks both:

- **Set drift**: every `.json` file matches an inline policy on the role,
  and vice versa.
- **Content drift**: per-policy document equality after JSON normalization.

```bash
# Local
./infrastructure/iam/check-drift.py
./infrastructure/iam/check-drift.py --role alpha-engine-executor-role
```

Exit code 0 = clean, 1 = drift detected, 2 = AWS CLI error or invalid
source JSON.

In CI, `.github/workflows/iam-drift-check.yml` runs the same script:

- On every PR that touches `infrastructure/iam/**` (catches code changes
  that forgot to apply, or applies that forgot to commit)
- Daily at 09:30 UTC (catches out-of-band manual IAM edits)
- Manually via `workflow_dispatch`

Auth uses OIDC via the `github-actions-iam-drift-check` role (read-only:
`iam:ListRolePolicies` + `iam:GetRolePolicy` on the codified roles only).

## When you add a new inline policy

1. Apply it to AWS first (e.g. via `aws iam put-role-policy ...`)
2. Save the JSON document to the matching directory
3. Commit the file with a description of why the grant was needed

## When you remove an inline policy

1. Delete the file from this directory
2. Run `aws iam delete-role-policy --role-name <role> --policy-name <policy>`
3. Commit the deletion

The flat-file approach is intentionally low-ceremony — if the blast radius
grows (cross-account, multiple roles per service, complex trust-policy
state), migrate to CloudFormation/Terraform.
