# agentcore-release-showcase

Showcase of the
[AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
GitHub Action: a customer-support agent on Amazon Bedrock AgentCore, released
through a live A/B test and gated by a code-based evaluator.

Two releases ran: a latency optimization was rolled back and a prompt fix was
promoted. The full write-up is on
[dev.to](https://dev.to/alvarog2491/releasing-and-evaluating-ai-agents-on-amazon-bedrock-agentcore-with-ab-tests-and-explainability-2hp9).

## Tech stack

| Layer | Technology |
|---|---|
| Agent | Python, LangGraph, LangChain AWS (`ChatBedrockConverse`) |
| Model | `openai.gpt-oss-20b` on Amazon Bedrock |
| Hosting | Amazon Bedrock AgentCore Runtime (ARM64 container in Amazon ECR) and an HTTP AgentCore Gateway |
| Evaluation | AgentCore online evaluations with a code-based evaluator on AWS Lambda (Python 3.13) |
| Observability | OpenTelemetry (ADOT), Amazon CloudWatch and AWS X-Ray |
| Infrastructure | Terraform (AWS provider 6.x), deployed to `eu-central-1` |
| CI/CD | GitHub Actions with OIDC and the AgentCore A/B Release Gate action |
| Tooling | uv workspace, Docker |

## Structure

```
.
├── src/
│   ├── agent/          # LangGraph agent with Amazon Bedrock, served by AgentCore Runtime
│   └── evaluators/     # Code-based evaluator (Lambda) that gates each release
├── infra/              # Terraform: runtime, gateway, evaluators, IAM roles
├── .github/workflows/  # Release workflow: build, A/B release gate, traffic
├── scripts/            # Traffic generation, test sessions, on-demand evaluations
└── docs/               # Source of the dev.to post
```

## Learn more

The [dev.to post](https://dev.to/alvarog2491/releasing-and-evaluating-ai-agents-on-amazon-bedrock-agentcore-with-ab-tests-and-explainability-2hp9)
covers the deployment, both release runs and the evaluator in detail. The
release gate itself lives in the
[AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
repository.
