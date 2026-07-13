# IAM policies (alpha-engine — module-specific roles)

Source-of-truth for the inline IAM policies on this repo's IAM roles.
Module-specific roles only — cross-cutting orchestration roles
(SF execution role, EventBridge cron role, GitHub Actions Lambda
deploy role) live in `alpha-engine-data/infrastructure/iam/` because
their grants are derived from code that lives there.

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
is the inline policy name on that role — EXCEPT two reserved filenames per
role dir, which extend coverage to non-inline axes:

- `trust-policy.json` — the role's assume-role (trust) policy document
- `managed-policies.json` — a JSON array of attached managed-policy ARNs

Both are opt-in per role (codify the file to start enforcing that axis;
absent ⇒ that axis is skipped for that role).

## Roles managed here

- **`alpha-engine-executor-role`** — assumed by the **trading** EC2 instance
  (`ae-trading`) AND, via the same `alpha-engine-executor-profile` instance
  profile, the ephemeral groom-dispatch spot boxes (`groom_spot_bootstrap.sh`
  — full groom runs and the standalone end-of-SF sweep box both launch under
  this profile). 10 inline policies (trading-scoped: S3, SES, SNS,
  CloudWatch, SSM read/send, EC2 spot, EOD Step Function; plus
  `alpha-engine-stepfunctions-diagnose`, added 2026-07-13 — read-only
  `states:ListExecutions`/`DescribeExecution`/`DescribeStateMachine` on the
  three trading-pipeline state machines, so the groom sweep box's
  deterministic `gate:*-sf` sweep (`gate_sf_run_sweep.py`, config#2397) can
  check whether a named pipeline has succeeded since a gate was applied,
  without a cross-role `AssumeRole` hop into the GHA-OIDC-only
  `saturday-sf-watch-role`). Trust policy
  (`ec2.amazonaws.com`) and managed attachment (`AmazonSSMManagedInstanceCore`)
  are codified via the reserved files. Until 2026-06-09 this role was also the
  instance profile for the dashboard box and carried dashboard/cyphering/mnemon
  grants — those were split out to `alpha-engine-dashboard-role` (see
  "Per-box role split" below).
- **`alpha-engine-dashboard-role`** — assumed by the **dashboard/monitoring**
  EC2 box (`i-09b539c844515d549`). Created 2026-06-09 to isolate the low-trust
  monitoring workload from the high-trust trading role. Shared by all four
  single-operator monitoring surfaces on the box: alpha-engine `console`/`live`
  Streamlit, the cyphering signal site, the box-health watchdog, and
  **robodashboard** (`portfolio.nousergon.ai`). robodashboard is an
  **intentional co-tenant** — it stays a separate repo (per the 2026-06-03
  keep-separate-product decision) but is treated as part of the nous-ergon
  shared nous-ergon trust domain, so it shares this role rather than getting its
  own. ⚠️ **Commercialization gate:** the moment robodashboard serves its first
  *external* customer (multi-tenant, others' brokerage data) it becomes a
  separate trust domain and MUST move to dedicated compute + its own role +
  its own bucket (ideally its own AWS account) — do NOT ship multi-tenant on
  this shared monitoring role. Policies: `alpha-engine-research-access`
  (read-all on `alpha-engine-research` + `PutObject` scoped to `dashboard/*`,
  `decision_artifacts/_calibration/*`, `decision_artifacts/_spotcheck/*`,
  `_alerts/*`, `robodashboard/*` — no Delete, no full-bucket write; right-sized
  2026-06-09), SSM read, SFN read, SNS, CloudWatch, cyphering SSM read,
  mnemon S3, trust (`ec2`), `AmazonSSMManagedInstanceCore`.
- **`github-actions-iam-drift-check`** — assumed by GitHub Actions via
  OIDC for the daily IAM-drift-check workflow. Single inline policy
  granting `iam:ListRolePolicies` + `iam:GetRolePolicy` + `iam:GetRole` +
  `iam:ListAttachedRolePolicies` (the last two added 2026-06-09 for the
  trust/managed coverage axes) scoped to every codified role across
  alpha-engine + alpha-engine-data + alpha-engine-predictor.
  Trust policy: `repo:nousergon/crucible-executor` + `repo:nousergon/nousergon-data`
  (main + pull_request); widened 2026-05-06 to support alpha-engine-data's
  drift-check workflow when the cross-cutting orchestration roles moved
  to that repo.

## Roles owned elsewhere

| Role | Home repo | Why there |
|---|---|---|
| `alpha-engine-step-functions-role` | `alpha-engine-data` | Grants reflect the Lambdas the SF JSON invokes + EC2 instances it SSMs + the trading instance it starts/stops — all defined in `alpha-engine-data/infrastructure/`. |
| `alpha-engine-eventbridge-sfn-role` | `alpha-engine-data` | Grants reflect which SFs the EventBridge cron rules target — same source repo. |
| `github-actions-lambda-deploy` | `alpha-engine-data` | Cross-cutting; assumed by Lambda deploy workflows in multiple repos. |
| `alpha-engine-predictor-role` | `alpha-engine-predictor` | Predictor Lambda's execution role. |

Each repo has its own `apply.sh` + `check-drift.py` scoped to its own
codified roles. The foreign-writer guard (`check-no-foreign-writers.py`)
in this directory scans every sibling repo for codified-role writes
that bypass the home repo's `apply.sh`, regardless of where the role
is codified.

## Coverage (what's codified + checked)

`apply.sh` and `check-drift.py` cover three axes per role:

1. **Inline policies** — every `<role>/<name>.json` (except the reserved
   filenames) ⇄ the role's inline policies.
2. **Trust policy** — `<role>/trust-policy.json` ⇄ `AssumeRolePolicyDocument`
   (opt-in; codified 2026-06-09 for all roles here).
3. **Managed attachments** — `<role>/managed-policies.json` (array of ARNs)
   ⇄ attached managed policies (opt-in; additive on apply — `apply.sh` WARNs
   on attached-but-uncodified ARNs and never auto-detaches).

Still out of scope: role **creation** (bootstrapped once via
`migrate-dashboard-role.sh` or by hand; thereafter `apply.sh` manages the
existing role's policies/trust/attachments).

## Per-box role split (2026-06-09)

The two EC2 boxes used to share ONE instance profile
(`alpha-engine-executor-profile` → `alpha-engine-executor-role`), making that
role a catch-all spanning four projects (executor + dashboard + cyphering +
mnemon). `migrate-dashboard-role.sh` splits them so the high-trust trading
role is isolated from the low-trust monitoring/cyphering workload:

```bash
./infrastructure/iam/migrate-dashboard-role.sh status         # show current state
./infrastructure/iam/migrate-dashboard-role.sh create         # additive: new role+profile+policies
./infrastructure/iam/migrate-dashboard-role.sh swap           # repoint the live dashboard box
#   ... verify dashboard + cyphering site + box-health alerts ...
./infrastructure/iam/migrate-dashboard-role.sh trim-executor  # drop dashboard-only grants from trading role
./infrastructure/iam/migrate-dashboard-role.sh rollback       # repoint box back (if swap misbehaves)
```

Each step accepts `--dry-run`. Deferred (P2): right-size `alpha-engine-dashboard-role`
further (drop the broad `research-bucket-write` for a scoped `_alerts`/`health`
write policy) via IAM Access Analyzer least-privilege generation after a soak.

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

## Drift detection (codified vs live AWS)

`check-drift.py` diffs the codified state against AWS for every role
directory under `infrastructure/iam/`, across all three coverage axes:

- **Set drift**: every inline `.json` file matches an inline policy on the
  role, and vice versa.
- **Content drift**: per-policy document equality after JSON normalization.
- **Trust drift**: `trust-policy.json` ⇄ live `AssumeRolePolicyDocument`.
- **Managed drift**: `managed-policies.json` ⇄ live attached managed ARNs.

```bash
# Local
./infrastructure/iam/check-drift.py
./infrastructure/iam/check-drift.py --role alpha-engine-executor-role
```

Exit code 0 = clean, 1 = drift detected, 2 = AWS CLI error or invalid
source JSON.

## Foreign-writer detection (multi-writer regressions)

`check-no-foreign-writers.py` enforces the **single-writer rule**: each
codified role must have exactly one writer (`apply.sh` in this repo).
Any deploy script in any sibling repo that calls `aws iam put-role-policy`
against a codified role name is a regression risk and fails the check.

This catches the regression class behind 4 IAM-clobber incidents in two
months (EB-SFN role 2026-04-21 + 2026-05-04 + 2026-05-06; SF role
2026-05-04 EOD + 2026-05-06 morning). All four had the same shape: a
codified policy with `apply.sh` as the sanctioned writer + a stale
inline `put-role-policy` block in `alpha-engine-data` deploy scripts.
Whichever ran last won.

```bash
# Local — scans this repo + all sibling alpha-engine-* repos that exist
./infrastructure/iam/check-no-foreign-writers.py

# Scope to a single repo
./infrastructure/iam/check-no-foreign-writers.py --repo ~/Development/alpha-engine-data
```

Exit code 0 = clean, 1 = foreign writer detected.

## CI integration

`.github/workflows/iam-drift-check.yml` runs both checks:

- **Drift check** — needs OIDC AWS read access. Compares codified to live.
- **Foreign-writers check** — pure source scan, clones every sibling
  alpha-engine-* repo and greps for `put-role-policy` against codified
  role names. No AWS auth needed.

Triggers: every PR touching `infrastructure/iam/**`, daily at 09:30 UTC,
manual `workflow_dispatch`.

Auth (drift-check only): OIDC via the `github-actions-iam-drift-check`
role (read-only: `iam:ListRolePolicies` + `iam:GetRolePolicy` + `iam:GetRole`
+ `iam:ListAttachedRolePolicies` on the codified roles).

## When you add a new inline policy

1. Apply it to AWS first (e.g. via `aws iam put-role-policy ...`)
2. Save the JSON document to the matching directory
3. Commit the file with a description of why the grant was needed

## When you remove an inline policy

1. Delete the file from this directory
2. Run `aws iam delete-role-policy --role-name <role> --policy-name <policy>`
3. Commit the deletion

The flat-file approach is intentionally low-ceremony. If the blast radius
grows (cross-account, multiple roles per service, complex trust-policy
state) and declarative IaC becomes worth it, fold IAM into the **existing
CloudFormation** (`alpha-engine-orchestration` stack) rather than introducing
Terraform — adding a third IaC tool alongside CFN + this flat-file layer
would create drift between tools, not reduce it.

<!-- ci-trigger after alpha-engine-data#172 merged -->
