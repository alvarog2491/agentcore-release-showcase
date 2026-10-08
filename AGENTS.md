# AGENTS.md

Guidance for coding agents working in this repository.

## Purpose

Showcase for the [AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
GitHub Action. The repo provides an agent plus the AWS infrastructure the action
requires; the action then deploys new agent images as A/B candidates and
promotes or rolls them back.

## Layout

- `src/agent/`: the agent (LangGraph `create_react_agent`, `ChatBedrockConverse`,
  `BedrockAgentCoreApp`). Entry point `main.py`, model setup in `model/load.py`.
  Has its own `pyproject.toml` and `Dockerfile`.
  The agent is a customer-support assistant for a fictitious online store. Its tools
  live in `tools/store.py` and share one fixed, in-memory dataset (orders,
  shipments, returns) with a fixed `TODAY`, so evaluation runs are
  deterministic. The tools chain together: `find_customer_orders` →
  `get_order_details` → `track_shipment` or `check_return_eligibility` →
  `create_return`. When adding tools, keep them in this domain, register them
  in `TOOLS`, and mention them in `DEFAULT_SYSTEM_PROMPT` in `main.py`. New
  packages must be added to `[tool.hatch.build.targets.wheel] packages`.
- `src/evaluators/support_workflow.py`: Lambda handler for the deterministic
  support-workflow evaluator (the release gate). Standard library only; not a
  uv workspace member and not in the agent image. Terraform zips and deploys
  only this file. Tests in `test_support_workflow.py`
  (`python -m unittest discover src/evaluators`).
- `infra/`: Terraform (AWS provider `~> 6.63`, needed for HTTP gateway targets), one file per concern:
  `ecr.tf`, `iam.tf`, `runtime.tf`, `gateway.tf`, `evaluations.tf`, `evaluators.tf`, `release.tf`
  (A/B test role and GitHub OIDC deploy role). Deployed to `eu-central-1`.
- The GitHub deploy role trusts the OIDC `sub` claim prefix in
  `var.github_oidc_sub_prefix`. This repo uses GitHub's immutable subjects
  (`repo:<owner>@<owner-id>/<repo>@<repo-id>:...`), not `repo:<owner>/<repo>:...`;
  read the real prefix with
  `gh api repos/<owner>/<repo>/actions/oidc/customization/sub`.
- The infrastructure is applied by hand with local Terraform state
  (`infra/terraform.tfstate`, gitignored), never from GitHub Actions; see the
  README.
- `scripts/traffic.sh`: sends customer questions through the Gateway (SigV4)
  so the A/B test has sessions to score. Used by the workflow's `traffic` job
  and for local smoke tests.
- `scripts/test_agent.py` and `scripts/evaluate.py`: multi-turn test sessions
  against the runtime, and on-demand evaluations of one session (downloads its
  spans from CloudWatch, calls `Evaluate` per evaluator, writes a report to
  `results/`, gitignored). Run with `uv run`; they only need boto3.
- `docs/devto-agentcore-release-and-evaluation.md`: the single dev.to post
  (release gate: a latency optimization rolled back, then a prompt fix promoted),
  written in the style of the AWS Machine Learning blog (prose and tables, no
  generated diagrams). Sections marked `TODO` wait for real release runs; never
  fill them with invented numbers.
- `README.md` holds the version 1 (baseline) and version 2 (latency edit,
  rolled back) prompts the post uses; the repository's `DEFAULT_SYSTEM_PROMPT`
  is version 3 (SKU fix, promoted).

## Tooling

- Python is managed with **uv** as a workspace: the root `pyproject.toml` lists
  `src/agent` as a member. There is exactly one `.venv` and one `uv.lock`, both
  at the repo root. Never create a venv or lockfile inside `src/agent`.
- Terraform manages its own providers (`terraform -chdir=infra init`). Don't use
  npm or any other package manager for it.

## Commands

- `uv sync`: install everything into the root venv.
- `uv run python -c "import main"` from `src/agent`: quick import check.
- `terraform -chdir=infra fmt` / `validate` / `plan`.
- `docker build --platform linux/arm64 -f src/agent/Dockerfile .`: build the
  agent image. The context **must be the repo root** (the image installs from
  the root `uv.lock` with `uv sync --package myagent`). The ignore file is
  `src/agent/Dockerfile.dockerignore`, an allowlist; keep it that way.

## Release workflow

`.github/workflows/release.yml` runs on pushes to `main` that touch the agent
(or manually):

1. `publish` builds the ARM64 image on an ARM runner and pushes it to ECR,
   tagged with the commit SHA.
2. `deploy` hands the image digest to the release gate with `step: auto`
   (observe and promote in one job, no manual approval). Showcase-sized test:
   900 s observation, 50/50 split.
3. `traffic` runs in parallel with `deploy` and calls `scripts/traffic.sh`
   for the observation window plus 10 minutes of setup time.

Account-specific values come from GitHub repository variables:
`AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `AGENTCORE_RUNTIME_ID`,
`AGENTCORE_GATEWAY_ID`, `AGENTCORE_EVALUATION_CONFIG_ID`,
`AGENTCORE_AB_TEST_ROLE_ARN`, and optionally `AB_TEST_DURATION_SECONDS`
(default 7200). Quality gates are hardcoded in the workflow's `quality-gates`
input and passed straight to the action. The action requires full evaluator
IDs (`<name>-<10-char suffix>`), not names; the suffix is generated by AWS, so
after creating or recreating the evaluator, update the ID in the workflow from
`terraform output support_workflow_evaluator_id`. With `require-significance`
on (the default), a candidate is promoted only if it is significantly better
than control; an equally good candidate is rolled back. Pin every third-party action to a full commit SHA with the
version in a comment.

## Infrastructure contract with the action

The action expects these to exist. Keep them intact when editing `infra/`:

- An AgentCore Runtime (`aws_bedrockagentcore_agent_runtime.agent`, HTTP protocol).
- A Runtime endpoint named `control` (`var.control_endpoint_name`), matching the
  action's `control-endpoint-name` input.
- A **dedicated HTTP Gateway**: `aws_bedrockagentcore_gateway` with **no**
  `protocol_type` (setting `MCP` breaks HTTP targets).
- A Gateway target with the same name as the control endpoint, using
  `target_configuration.http.agentcore_runtime` with `qualifier` = the endpoint name.
- A Runtime execution role with the X-Ray permissions for online evaluation.

The action changes resources at release time. Terraform must not fight it:

- The Runtime's `agent_runtime_artifact` (image) and the control endpoint's
  `agent_runtime_version` are in `lifecycle.ignore_changes`. Keep them there.
- The action creates the `treatment` endpoint and gateway target itself. Don't
  define them in Terraform.

The GitHub OIDC provider already exists in the account; `release.tf` reads it
with a data source instead of creating it.

## Evaluations

Releases are gated by one deterministic, code-based evaluator; no LLM judges
and no built-in evaluators in the online config. One LLM-as-a-judge evaluator
exists only for on-demand evaluation (see the end of this section).

- `showcase_agent_support_workflow` (TRACE level) is
  `src/evaluators/support_workflow.py` on Lambda. Each agent turn scores 1.0
  (PASS) unless it breaks a rule of the store's support workflow, else 0.0
  (FAIL): it states order facts with no tool call in the session; it contains
  an order ID, tracking number or RMA number found in no customer message or
  tool result; it asks the customer for a SKU without having called
  `get_order_details`; or it calls `create_return` without an earlier
  `check_return_eligibility` for the same order and SKU. Rules read the
  `execute_tool` spans (tool name, `gen_ai.tool.call.arguments`,
  `gen_ai.tool.call.result`) and the `LangGraph.workflow` span (customer
  message, final answer) emitted by `opentelemetry-instrumentation-langchain`.
  The model writes non-ASCII hyphens (`ORD\u20111001`); keep normalizing them.
- The rules encode the store's workflow, not a specific release. When changing
  them, rerun the unit tests and check the current prompt still scores ~1.0.
- The workflow gate `{"showcase_agent_support_workflow-<suffix>": 0.9}` means
  at least 90% of treatment turns follow every rule.
- Lambda contract: input `evaluationInput.sessionSpans` (ADOT span dicts) plus
  `evaluationTarget.traceIds`; output `{label, value, explanation}` or
  `{errorCode, errorMessage}`. Keep the handler dependency-free.
- `aws_bedrockagentcore_online_evaluation_config.control` is the gate's template.
  Its log group (`/aws/bedrock-agentcore/runtimes/<runtime_id>-control`) and
  service name (`<agent_name>.control`) must contain the control endpoint name,
  which the gate swaps for `treatment`. `agent_name` must not contain it.
- Evaluators referenced by an enabled online evaluation config are locked,
  and Terraform can't change the evaluators of an enabled config in one
  apply. To add or remove an evaluator: `aws bedrock-agentcore-control
  update-online-evaluation-config --online-evaluation-config-id <id>
  --execution-status DISABLED`, `terraform -chdir=infra apply`, then the same
  command with `ENABLED` (the action needs an enabled template). Changing
  only the Lambda code is fine.
- CloudWatch Transaction Search must be enabled in the account for online
  evaluation to see agent spans.
- `infra/evaluators.tf` defines `aws_bedrockagentcore_evaluator.order_grounding`,
  a TRACE-level LLM-as-a-judge evaluator (are order facts in the answer backed
  by tool results?) with the `{context}` and `{assistant_turn}` placeholders,
  the store policy (keep it in sync with `tools/store.py`) and a
  Yes/Partially/No (1.0/0.5/0.0) scale. The judge is `var.judge_model_id`.
  Do not add it to the online config without deciding to gate releases on
  it: that locks it and adds judge cost to every scored session.

## Conventions

- Runtime names allow only letters, digits and underscores (`showcase_agent`).
  Gateway and target names use hyphens.
- Keep the Bedrock model ID in sync between `infra/variables.tf`
  (`bedrock_model_id`) and `src/agent/model/load.py` (`MODEL_ID`).
- Never commit or push without the user's explicit approval.
