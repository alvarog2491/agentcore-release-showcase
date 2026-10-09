# agentcore-release-showcase

Showcase of the
[AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
GitHub Action. A customer-support agent on Amazon Bedrock AgentCore is
released through a live A/B test, and a deterministic, code-based evaluator
decides whether each new version is promoted or rolled back.

The walkthrough is published on dev.to:
[Gating AI agent releases on Amazon Bedrock AgentCore with A/B tests and a code-based evaluator](docs/devto-agentcore-release-and-evaluation.md).

## Results

Two releases run against the same baseline. A latency optimization that skips
the return eligibility check is rolled back, and a prompt fix that reads SKUs
from the order is promoted.

| Release | Control | Treatment | p-value | Result |
|---|---|---|---|---|
| Latency optimization | 0.924 | 0.649 | 6.3e-9 | Rolled back |
| SKU fix | 0.914 | 1.00 | 0.0047 | Promoted |

## Contents

| Path | Contents |
|---|---|
| `src/agent/` | The agent: LangGraph with Amazon Bedrock, served by AgentCore Runtime |
| `src/evaluators/` | The code-based evaluator that scores each turn against the store's support workflow |
| `infra/` | Terraform for the runtime, gateway, evaluators and IAM roles the action uses |
| `.github/workflows/` | The release workflow: build, A/B release gate and traffic |
| `scripts/` | Traffic generation, test sessions and on-demand evaluations |
| `docs/` | The dev.to post |
