# agentcore-release-showcase

A general-purpose showcase of how to release and evaluate an Amazon Bedrock
AgentCore agent with the
[AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
GitHub Action. Use it as a template: replace the agent, the change you release
and the evaluators with your own, and the same flow applies.

- **Release.** A new agent image is promoted only when it scores at least as
  well as the current version in a live A/B test.
- **Evaluate.** The released agent is evaluated on demand for helpfulness,
  business accuracy and explainability, following the three-layer approach from
  the AWS Machine Learning blog post
  [Evaluating multi-agent systems for explainability and helpfulness with Amazon Bedrock AgentCore](https://aws.amazon.com/blogs/machine-learning/evaluating-multi-agent-systems-for-explainability-and-helpfulness-with-amazon-bedrock-agentcore/).

The example agent is a customer-support assistant for a fictitious online
store. The full walkthrough, with results from real runs, is in
[`docs/devto-agentcore-release-and-evaluation.md`](docs/devto-agentcore-release-and-evaluation.md).

## Layout

| Path | What it is |
|---|---|
| `src/agent/` | The agent: LangGraph + Bedrock, served with the AgentCore Runtime SDK. |
| `src/evaluators/` | The release gate's code-based evaluator (Lambda). |
| `infra/` | Terraform for the AWS resources the action needs, plus custom evaluators. |
| `scripts/` | Traffic generation, test sessions and on-demand evaluations. |
| `docs/` | The dev.to post: releasing the agent with the A/B release gate and evaluating it. |

## Infrastructure

`infra/` creates the prerequisites the action expects:

- **ECR repository** for the agent images.
- **AgentCore Runtime** running the agent container (HTTP protocol).
- **`control` Runtime endpoint**: the stable version serving traffic.
- **AgentCore Gateway** (HTTP, IAM auth) with a **`control` target** that routes
  to the control endpoint. During a release, the action adds a `treatment`
  endpoint and target and splits Gateway traffic between them.
- **A deterministic tool-usage evaluator**: a Lambda
  (`src/evaluators/tool_usage.py`) that scores each agent turn 1 if it called a
  tool and 0 if not, plus the **online evaluation configuration** the action
  uses to score control and treatment with it.
- **Six custom LLM-as-a-judge evaluators** (`infra/evaluators.tf`) for on-demand
  evaluation. See [Evaluations](#evaluations).

Terraform bootstraps these resources. After that, the action owns the Runtime
image and the endpoint versions, and Terraform ignores changes to them.

The Terraform outputs `runtime_id`, `gateway_id`, `aws_region` and
`evaluation_config_id` are the action's inputs.

## Deploying the infrastructure (manual, one time)

The infrastructure is deployed by hand; agent releases run in GitHub Actions.
Requires Terraform >= 1.14, Docker, uv and AWS credentials for the target
account (`eu-central-1` by default).

1. Create the ECR repository, then push a first image tagged `bootstrap`
   (the Runtime is created from it):

   ```bash
   terraform -chdir=infra init
   terraform -chdir=infra apply -target=aws_ecr_repository.agent
   REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
   aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin "${REPO%%/*}"
   docker buildx build --platform linux/arm64 --provenance=false -f src/agent/Dockerfile -t "$REPO:bootstrap" --push .
   ```

2. Create everything else:

   ```bash
   terraform -chdir=infra apply
   ```

3. Copy the outputs into the GitHub repository variables used by the release
   workflow (`AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `AGENTCORE_RUNTIME_ID`,
   `AGENTCORE_GATEWAY_ID`, `AGENTCORE_EVALUATION_CONFIG_ID`,
   `AGENTCORE_AB_TEST_ROLE_ARN`), create a `production` environment, and set
   the evaluator ID from `terraform output tool_usage_evaluator_id` in the
   workflow's `quality-gates`.

4. Set `github_oidc_sub_prefix` to your repository's OIDC subject prefix if you
   use a fork:

   ```bash
   gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
   ```

## Release workflow

[`.github/workflows/release.yml`](.github/workflows/release.yml) runs on every
push to `main` that changes the agent. It publishes a new ARM64 image to ECR,
then the release gate A/B tests it against the current version. The gate
promotes the image when the agent used its tools in at least 80% of treatment
turns, and rolls it back otherwise.

The action only observes traffic. `scripts/traffic.sh` plays the customers
during the observation period; in production, your users generate the traffic.

## Evaluations

`infra/evaluators.tf` creates six TRACE-level LLM-as-a-judge evaluators that
complete the three layers:

| Layer | Evaluators |
|---|---|
| 1. General quality | Built-in: `Builtin.Helpfulness`, `Builtin.ToolSelectionAccuracy`, `Builtin.Faithfulness`, `Builtin.GoalSuccessRate` |
| 2. Business accuracy | `return_policy`, `order_grounding` |
| 3. Explainability | `decision_rationale`, `evidence_attribution`, `policy_reasoning`, `assumption_disclosure` |

They run on demand only, outside the online evaluation config, so releases use
the tool-usage evaluator alone.

```bash
# Multi-turn test sessions (orders, shipping, returns); prints the session IDs
uv run scripts/test_agent.py --runtime-arn "$(terraform -chdir=infra output -raw runtime_arn)"

# Wait 3-5 minutes for the spans to reach CloudWatch, then:
uv run scripts/evaluate.py --runtime-id "$(terraform -chdir=infra output -raw runtime_id)" \
  --session-id <session-id> \
  --evaluators showcase_agent_return_policy,Builtin.Helpfulness,Builtin.ToolSelectionAccuracy
```

Reports are saved under `results/` (gitignored).

## Cleaning up

1. Delete what the action created outside Terraform: the stopped A/B tests of
   the gateway (`DeleteABTest` API), the `treatment` gateway target and the
   `treatment` runtime endpoint.
2. Run `terraform -chdir=infra destroy`.
3. Delete the log groups AgentCore created itself:
   `/aws/bedrock-agentcore/evaluations/results/showcase_agent_*` and the runtime's
   `DEFAULT` endpoint log group.
