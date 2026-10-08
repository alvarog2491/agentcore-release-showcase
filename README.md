# agentcore-release-showcase

A general-purpose showcase of how to release and evaluate an Amazon Bedrock
AgentCore agent with the
[AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
GitHub Action. Use it as a template: replace the agent, the change you release
and the evaluators with your own, and the same flow applies.

- **Release.** A new agent image is promoted only when it scores better than
  the current version in a live A/B test, and rolled back otherwise.
- **Gate.** A deterministic, code-based evaluator scores every agent turn
  against the store's support workflow: look facts up, invent no order,
  tracking or return numbers, never ask the customer for a SKU, and check
  eligibility before opening a return.

The example agent is a customer-support assistant for a fictitious online
store. The walkthrough releases a latency optimization that skips the
eligibility check (rolled back), then a prompt fix (promoted). The full
walkthrough, with results from real runs, is in
[`docs/devto-agentcore-release-and-evaluation.md`](docs/devto-agentcore-release-and-evaluation.md).

## Layout

| Path | What it is |
|---|---|
| `src/agent/` | The agent: LangGraph + Bedrock, served with the AgentCore Runtime SDK. |
| `src/evaluators/` | The release gate's code-based evaluator (Lambda) and its tests. |
| `infra/` | Terraform for the AWS resources the action needs, plus the evaluators. |
| `scripts/` | Traffic generation, test sessions and on-demand evaluations. |
| `docs/` | The dev.to post: releasing the agent with the A/B release gate. |

## Infrastructure

`infra/` creates the prerequisites the action expects:

- **ECR repository** for the agent images.
- **AgentCore Runtime** running the agent container (HTTP protocol).
- **`control` Runtime endpoint**: the stable version serving traffic.
- **AgentCore Gateway** (HTTP, IAM auth) with a **`control` target** that routes
  to the control endpoint. During a release, the action adds a `treatment`
  endpoint and target and splits Gateway traffic between them.
- **A deterministic support-workflow evaluator**: a Lambda
  (`src/evaluators/support_workflow.py`) that scores each agent turn 1 if it
  follows the store's support workflow and 0 if it breaks a rule, plus the
  **online evaluation configuration** the action uses to score control and
  treatment with it.
- **One LLM-as-a-judge evaluator** (`infra/evaluators.tf`) for on-demand
  evaluation of the facts the rules can't check. See [Evaluations](#evaluations).

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
   the evaluator ID from `terraform output support_workflow_evaluator_id` in
   the workflow's `quality-gates`.

4. Set `github_oidc_sub_prefix` to your repository's OIDC subject prefix if you
   use a fork:

   ```bash
   gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
   ```

## Release workflow

[`.github/workflows/release.yml`](.github/workflows/release.yml) runs on every
push to `main` that changes the agent. It publishes a new ARM64 image to ECR,
then the release gate A/B tests it against the current version. The gate
promotes the image when at least 90% of treatment turns follow the support
workflow and the treatment scores significantly better than the control, and
rolls it back otherwise.

The action only observes traffic. `scripts/traffic.sh` plays the customers
during the observation period; in production, your users generate the traffic.

## The three agent versions

The walkthrough uses three versions of `DEFAULT_SYSTEM_PROMPT` in
`src/agent/main.py`. The repository contains version 3.

Version 1, the bootstrap image. It asks customers for SKUs in some return
requests:

```python
DEFAULT_SYSTEM_PROMPT = """
You are a customer-support assistant for an online electronics store.
Always use the available tools to answer — never guess order, shipping or
return information from your own knowledge.
- To find a customer's orders from their email, use find_customer_orders.
- To see items, SKUs, prices and dates of an order, use get_order_details.
- For "where is my order" questions, use track_shipment.
- Before promising a return, use check_return_eligibility; only call
  create_return when the customer asks for it and the item is eligible.
If you are missing an email, order ID or item, ask the customer for it.
"""
```

Version 2, a latency optimization of version 1 that the gate rolls back. It
opens returns without checking eligibility first:

```python
DEFAULT_SYSTEM_PROMPT = """
You are a customer-support assistant for an online electronics store.
Customers hate waiting: keep replies short and use as few tool calls as
possible.
- To find a customer's orders from their email, use find_customer_orders.
- To see items, SKUs, prices and dates of an order, use get_order_details.
- For "where is my order" questions, use track_shipment.
- When the customer asks for a return, call create_return directly: it
  checks eligibility itself, so check_return_eligibility is an extra step.
If you are missing an email, order ID or item, ask the customer for it.
"""
```

Version 3, in the repository, reads SKUs from the order instead of asking the
customer for them. The gate promotes it over version 1.

## Evaluations

The release gate uses one evaluator, `showcase_agent_support_workflow`
(TRACE level). A turn scores 0 when it breaks any of these rules:

| Rule | A turn fails when |
|---|---|
| Look it up | It states order facts and no tool was called in the session |
| No invented identifiers | It contains an order ID, tracking number or RMA number that no customer message or tool result contains |
| Don't make the customer do the lookup | It asks the customer for a SKU without having read the order |
| Check before acting | It calls `create_return` without an earlier `check_return_eligibility` for the same item |

```bash
python -m unittest discover src/evaluators
```

`infra/evaluators.tf` adds `showcase_agent_order_grounding`, an
LLM-as-a-judge evaluator that checks whether every order fact in an answer
(dates, prices, statuses) comes from a tool result. It runs on demand only,
outside the online evaluation config:

```bash
# Multi-turn test sessions (orders, shipping, returns); prints the session IDs
uv run scripts/test_agent.py --runtime-arn "$(terraform -chdir=infra output -raw runtime_arn)"

# Wait 3-5 minutes for the spans to reach CloudWatch, then:
uv run scripts/evaluate.py --runtime-id "$(terraform -chdir=infra output -raw runtime_id)" \
  --session-id <session-id> \
  --evaluators showcase_agent_order_grounding,Builtin.Faithfulness
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
