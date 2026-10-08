---
title: "Gating AI agent releases on Amazon Bedrock AgentCore with A/B tests and a code-based evaluator"
published: false
description: "Promote a new agent version only after it beats the current one on live traffic, and roll back a plausible-looking regression automatically, with Amazon Bedrock AgentCore and a deterministic evaluator."
tags: aws, ai, githubactions, devops
---

A new version of an AI agent, with a new system prompt, model, or architecture, should reach every user only after it scores at least as well as the current version on real traffic. The hard part is not the deployment. It is deciding, automatically, whether the new version is better, and catching the change that looks harmless in review and still makes the agent worse.

Amazon Bedrock AgentCore provides the building blocks. AgentCore Runtime hosts multiple versions of an agent behind named endpoints, AgentCore Gateway splits traffic between them with an A/B test, and AgentCore Evaluations scores the agent's OpenTelemetry traces with built-in, custom LLM-as-a-judge or code-based evaluators, either continuously (online) or whenever you call the `Evaluate` API (on demand).

In this post, we use the [AgentCore A/B Release Gate](https://github.com/alvarog2491/agentcore-ab-release-gate) GitHub Action to release two versions of a customer-support agent. The action deploys each new container image next to the current version, splits live traffic between them, and promotes the new version only when it passes the quality gates you define. The gate is a deterministic, code-based evaluator that encodes the store's support workflow as rules. The first release is a latency optimization that reads well in a code review, breaks the workflow rules, and is rolled back. The second release fixes a real weakness of the agent and is promoted.

The showcase is a general-purpose scenario built to show how the action works from start to finish, so that you can apply it in your own work. The customer-support agent, its system prompts and its evaluator are examples. Replace them with your agent, the change you want to release, and the rules that matter in your domain, and the same release flow applies.

All logs and numbers in this post come from real runs.

{% embed https://github.com/alvarog2491/agentcore-release-showcase %}

## Solution overview

For this post, we use a customer-support assistant for a fictitious online electronics store. Customers ask the assistant which orders they have, where a parcel is, and whether they can return an item. The store has a simple return policy: an item can be returned within 30 days of delivery, final-sale items such as gift cards cannot be returned, and orders that haven't been delivered can't be returned yet.

The store also has a support workflow that every answer must follow: look the facts up instead of guessing them, never invent an order, tracking or return number, never ask the customer for information the agent can look up itself, and check that an item is eligible before opening a return for it. A code-based evaluator scores every turn of the agent against these rules, and the release gate promotes a new version only when it follows them at least as well as the current one.

## The agent

The agent serves the store's customers. A customer writes in natural language, for example "Hi, I'm ana@example.com. Where is my keyboard?", and the agent looks up the customer's data, answers with the facts it found, and opens a return when the customer asks for one and the store's policy allows it. It keeps the conversation of each session, so follow-up questions such as "What's the tracking number?" work.

The agent uses LangGraph's `create_react_agent` with the `openai.gpt-oss-20b-1:0` model on Amazon Bedrock and runs on AgentCore Runtime inside `BedrockAgentCoreApp`. It receives `{"prompt": "..."}` and answers with `{"result": "..."}`.

The agent has five tools. They read and write a small, fixed dataset in memory (two customers, four orders, their shipments and returns) with a fixed "today" of 2026-09-27, so every run sees the same facts and an invented answer is easy to spot:

| Tool | Input | What it returns |
|---|---|---|
| `find_customer_orders` | Customer email | The customer's order IDs, dates and statuses |
| `get_order_details` | Order ID | Items, SKUs, prices, total, status and delivery date |
| `track_shipment` | Order ID | Carrier, tracking number, delivery estimate and tracking events |
| `check_return_eligibility` | Order ID and SKU | Whether the item can be returned under the store policy, and until when |
| `create_return` | Order ID, SKU and reason | A new RMA number and the refund steps, only for an eligible item |

The tools chain together. A typical conversation goes from `find_customer_orders` to `get_order_details`, and then to `track_shipment` for a delivery question, or to `check_return_eligibility` and `create_return` for a return. The return policy lives in the tools: `check_return_eligibility` and `create_return` refuse final-sale items, undelivered orders and items outside the 30-day window.

The `opentelemetry-instrumentation-langchain` library records every model call and tool call as OpenTelemetry spans (tool calls as `execute_tool` spans), and AgentCore sends them to Amazon CloudWatch. All the evaluators in this post read these spans.

The following code shows the entrypoint in `src/agent/main.py`:

```python
@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    # Define the agent using create_react_agent (checkpointer is shared across invocations)
    graph = create_react_agent(
        get_or_create_model(),
        tools=TOOLS,
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )

    # Process the user prompt
    prompt = payload.get("prompt", "What can you help me with?")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    session_id = getattr(context, "session_id", "default-session")
    touch_thread(session_id)
    log.info(f"Agent input: {prompt}")

    # Run the agent (checkpointer auto-loads/saves history per session)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": session_id}},
    )

    # Return result
    # .text drops gpt-oss reasoning blocks and keeps only the answer text
    output = result["messages"][-1].text
    log.info(f"Agent output: {output}")
    return {"result": output}
```

### Three versions of the agent

In this post, the agent goes through three versions that differ only in `DEFAULT_SYSTEM_PROMPT`.

Version 1 is the version that serves users when the post starts. Its prompt tells the agent to use its tools and to ask the customer for anything that is missing:

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

Version 1 looks reasonable, and it answers order and shipping questions correctly. It has one weakness that shows up in return requests: `check_return_eligibility` and `create_return` need the item's SKU, and the last line tells the agent to ask for a missing item. In some return requests, the agent asks the customer for the SKU ("Could you tell me the SKU of the headphones in ORD-1001?") instead of reading it from the order. Customers don't know SKUs.

Version 2 is a latency optimization on top of version 1. Each tool call is one more model round trip, and `create_return` already refuses ineligible items, so the separate eligibility check looks redundant:

```diff
 You are a customer-support assistant for an online electronics store.
-Always use the available tools to answer — never guess order, shipping or
-return information from your own knowledge.
+Customers hate waiting: keep replies short and use as few tool calls as
+possible.
 - To find a customer's orders from their email, use find_customer_orders.
 - To see items, SKUs, prices and dates of an order, use get_order_details.
 - For "where is my order" questions, use track_shipment.
-- Before promising a return, use check_return_eligibility; only call
-  create_return when the customer asks for it and the item is eligible.
+- When the customer asks for a return, call create_return directly: it
+  checks eligibility itself, so check_return_eligibility is an extra step.
 If you are missing an email, order ID or item, ask the customer for it.
```

The change is easy to approve in a code review, and it saves a round trip per return. It also opens returns without telling the customer the conditions first (the deadline, the refund amount, whether the item qualifies at all), which the store's workflow requires, and the shorter prompt makes the agent ask for SKUs even more often.

Version 3, the one in the repository, fixes the weakness of version 1 instead. It tells the agent where SKUs come from and stops it from asking for them:

```diff
 - For "where is my order" questions, use track_shipment.
+- Returns need the item's SKU: read it with get_order_details, never ask the
+  customer for it.
 - Before promising a return, use check_return_eligibility; only call
   create_return when the customer asks for it and the item is eligible.
-If you are missing an email, order ID or item, ask the customer for it.
+If you are missing an email or order ID, ask the customer for it.
```

The prompts of versions 1 and 2 are in the repository's README.

Before releasing anything, we ran each version locally against the same model, with the 12 questions of the traffic script, four times each (48 single-turn sessions per version), and scored every answer with the release gate's rules (described in the next sections). The following table shows the results:

| Version | Scored turns that follow the workflow | Failed requests | Main failure |
|---|---|---|---|
| 1 | 42 of 48 (0.88) | 0 | Asks the customer for a SKU (6 turns) |
| 2 | 29 of 47 (0.62) | 1 | Asks for a SKU (11 turns) or calls `create_return` without `check_return_eligibility` (7 turns) |
| 3 | 48 of 48 (1.00) | 0 | None |

The failed request is a quirk of `openai.gpt-oss-20b-1:0`, which sometimes returns a tool call with an invalid tool name that the Converse API rejects. In the deployed agent, those requests fail and are not scored, so the gate doesn't see them. The differences in the table are large enough to measure in a 15-minute A/B test with enough traffic, which is what the release gate needs.

## Solution components

The solution uses the following components:

- **AgentCore Runtime** runs the agent container. Each image update creates a new runtime version.
- **Runtime endpoints** are names that point to one version. The `control` endpoint serves users. The action creates a `treatment` endpoint for the candidate version.
- **AgentCore Gateway** is the single entry point. It has one HTTP target per endpoint, and the A/B test splits traffic between the targets, per session.
- **AgentCore Evaluations** reads the agent's OpenTelemetry spans from Amazon CloudWatch. Online evaluation scores every session with a code-based evaluator (an AWS Lambda function) for the release gate. On-demand evaluation scores selected sessions with an LLM-as-a-judge evaluator for the facts that rules can't check.
- **GitHub Actions** builds the image, runs the release gate and generates traffic. It deploys through an IAM role assumed with OpenID Connect (OIDC), so there are no AWS keys in the repository.

## How the release gate works

The action expects an AgentCore Runtime with a `control` endpoint and a dedicated HTTP gateway in front of it. For every release, the action performs the following steps:

1. **Prepare.** It deploys the new image as a new runtime version, points a `treatment` endpoint to it, adds a `treatment` target to the gateway, and copies the online evaluation configuration once per variant.
2. **Observe.** It starts an A/B test on the gateway, so every new session goes either to control or to treatment, and waits while online evaluation scores the sessions of both variants.
3. **Decide.** For every evaluator in the quality gates, it checks that the treatment mean reaches the threshold, that it is not lower than the control mean, and that the difference is statistically significant (p < 0.05 by default).
4. **Promote or roll back.** If all gates pass, both endpoints move to the new version. If any gate fails, the job times out, the job is cancelled or AWS returns an error, both endpoints stay on (or return to) the previous version.

You can turn off the significance check when you want to gate on the threshold alone. With the check on, as in this post, a candidate is promoted only when it is measurably better than the current version. A candidate that is as good as the current version, but not better, is rolled back. That is why the version we want to promote, version 3, is a real improvement over version 1, and not a refactoring.

## The release gate evaluator

AgentCore Evaluations offers three kinds of evaluators:

| Evaluator type | How it scores | When to use it |
|---|---|---|
| Built-in | Pre-defined LLM-as-a-judge prompts such as `Builtin.Helpfulness` or `Builtin.Correctness` | General response quality with no setup |
| Custom LLM-as-a-judge | Your own instructions and rating scale, scored by a model you choose | Domain-specific checks that need judgment |
| Code-based | An AWS Lambda function that returns a score, label and explanation | Deterministic checks that code can decide |

Evaluators work at three levels: a whole session, a single turn (trace) or an individual tool call.

The release gate needs a score it can compare across variants. We use a code-based evaluator for it. It always gives the same score for the same trace, it costs a few milliseconds of Lambda time per call, and it returns a number, which the A/B statistics need. Its rules come from the store's support workflow, not from the change being released, so the same evaluator gates every future release.

The evaluator works at the TRACE level: it scores each turn of a session with the session's earlier turns as context. A turn scores 1.0 (PASS) when it breaks none of the following rules, and 0.0 (FAIL) when it breaks at least one:

| Rule | A turn fails when |
|---|---|
| Look it up | The answer states order facts (an order, tracking or return number, or a date) and no tool was called in the session |
| No invented identifiers | The answer contains an order ID, tracking number or RMA number that appears in no customer message and no tool result |
| Don't make the customer do the lookup | The answer asks the customer for a SKU, and the agent didn't read the order with `get_order_details` |
| Check before acting | `create_return` is called for an item without an earlier `check_return_eligibility` for the same order and SKU |

All four rules read only the spans that the agent already emits. The `opentelemetry-instrumentation-langchain` library records each tool call as an `execute_tool` span with the tool name, its arguments (`gen_ai.tool.call.arguments`) and its result (`gen_ai.tool.call.result`). The `LangGraph.workflow` span of each turn holds the customer's message and the agent's messages, including the final answer.

The following function in `src/evaluators/support_workflow.py` applies the rules to one turn. It uses only the Python standard library:

```python
def check_turn(customer_messages: list[str], tool_calls: list[dict], answer: str, turn_tool_calls: int) -> list[str]:
    """Return the workflow rules this turn breaks (empty when it passes).

    customer_messages: the customer's messages in the session up to this turn.
    tool_calls: the session's tool calls up to and including this turn, in
        order, as {"name", "inputs", "result"}.
    answer: the agent's final answer for this turn.
    turn_tool_calls: how many of tool_calls belong to this turn.
    """
    violations = []
    answer = _normalize(answer)
    mentioned = _identifiers(answer)
    from_customer = set()
    for message in customer_messages:
        from_customer |= _identifiers(message)

    # Repeating the customer's own order ID (for example, when asking for an
    # email) is not a fact the agent looked up.
    if (mentioned - from_customer or _DATE.search(answer)) and not tool_calls:
        violations.append("The answer states order facts but no tool was called to look them up.")

    known = set(from_customer)
    for call in tool_calls:
        known |= _identifiers(call.get("result", ""))
    invented = sorted(mentioned - known)
    if invented:
        violations.append(f"The answer contains identifiers no customer message or tool result supports: {', '.join(invented)}.")

    if _ASKS_FOR_SKU.search(answer) and not any(call.get("name") == "get_order_details" for call in tool_calls):
        violations.append("The agent asked the customer for a SKU instead of reading the order with get_order_details.")

    checked = set()
    for index, call in enumerate(tool_calls):
        inputs = call.get("inputs") or {}
        key = (_normalize(str(inputs.get("order_id", ""))).strip().upper(), str(inputs.get("sku", "")).strip().upper())
        if call.get("name") == "check_return_eligibility":
            checked.add(key)
        elif call.get("name") == "create_return" and key not in checked:
            if index >= len(tool_calls) - turn_tool_calls:
                violations.append(f"create_return was called for {key[0]} {key[1]} without check_return_eligibility first.")

    return violations
```

The rest of the file turns the spans into turns and calls `check_turn` for the turn under evaluation:

```python
def handler(event, context=None):
    spans = (event.get("evaluationInput") or {}).get("sessionSpans") or []
    trace_ids = (event.get("evaluationTarget") or {}).get("traceIds") or []
    turns = _turns(spans)

    # TRACE level scores one turn; without a target (SESSION level) score the last one.
    candidates = [i for i, t in enumerate(turns) if t["answer"] is not None and (not trace_ids or t["trace_id"] in trace_ids)]
    target = candidates[-1] if candidates else None
    if target is None:
        return {
            "errorCode": "NO_SPANS",
            "errorMessage": f"No agent turn found for trace(s) {trace_ids or 'in session'}.",
        }

    history = turns[: target + 1]
    turn = turns[target]
    violations = check_turn(
        customer_messages=[t["customer"] for t in history],
        tool_calls=[call for t in history for call in t["tool_calls"]],
        answer=turn["answer"],
        turn_tool_calls=len(turn["tool_calls"]),
    )
    if violations:
        return {"label": "FAIL", "value": 0.0, "explanation": " ".join(violations)}

    names = ", ".join(call["name"] for call in turn["tool_calls"]) or "none"
    return {"label": "PASS", "value": 1.0, "explanation": f"The turn follows the support workflow (tool calls: {names})."}
```

AgentCore calls the function with the spans of the session in `evaluationInput.sessionSpans` and the trace to score in `evaluationTarget.traceIds`. The function returns `label` and `value`, plus an optional `explanation`, or `errorCode` and `errorMessage` when there is nothing to score. A failed turn's explanation names the rule it broke, for example `create_return was called for ORD-1001 HD-200 without check_return_eligibility first.`

Two details matter when you write rules on model output:

- **Normalize the text.** The model often writes `ORD‑1001` with a non-breaking hyphen (U+2011) instead of `-`, so the evaluator normalizes hyphens before it matches identifiers.
- **Allow the agent to repeat what the customer said.** Asking "Could you share the email for order ORD-1004?" repeats the customer's order ID. It is not a fact the agent looked up, so the "look it up" rule ignores identifiers from customer messages.

We tested the evaluator in two ways before deploying it. Unit tests in `src/evaluators/test_support_workflow.py` cover each rule and the span parsing (`python -m unittest discover src/evaluators`). Then we passed the spans of real sessions, downloaded from CloudWatch, to the handler, and ran the prompt comparison shown earlier, which uses the same `check_turn` function.

The following Terraform registers the evaluator and the online evaluation configuration that the action uses as a template (from `infra/evaluations.tf`):

```hcl
resource "aws_bedrockagentcore_evaluator" "support_workflow" {
  evaluator_name = local.support_workflow_evaluator_name
  description    = "Deterministic: 1.0 if the turn follows the support workflow (lookups, grounded IDs, eligibility before returns), else 0.0."
  level          = "TRACE"

  evaluator_config {
    code_based {
      lambda_config {
        lambda_arn                = aws_lambda_function.support_workflow_evaluator.arn
        lambda_timeout_in_seconds = 30
      }
    }
  }

  depends_on = [aws_lambda_permission.agentcore_evaluations]
}

# --- Online evaluation config (release gate template) ---

# Runtime log groups are created by AgentCore on the first invocation.
# Pre-create the control one so the evaluation config can point at it
# from day one and so it gets a retention period.
resource "aws_cloudwatch_log_group" "control_runtime" {
  name              = "/aws/bedrock-agentcore/runtimes/${aws_bedrockagentcore_agent_runtime.agent.agent_runtime_id}-${var.control_endpoint_name}"
  retention_in_days = 30
}

# Template for the release gate (action input evaluation-config-id). The gate
# copies it per variant, replacing the control endpoint name with "treatment"
# in the service and log group names below, so both must contain it.
resource "aws_bedrockagentcore_online_evaluation_config" "control" {
  online_evaluation_config_name = "${var.agent_name}_${var.control_endpoint_name}_eval"
  description                   = "Evaluation of the control endpoint; template for A/B releases"
  enable_on_create              = true
  evaluation_execution_role_arn = aws_iam_role.evaluation.arn

  data_source_config {
    cloudwatch_logs {
      log_group_names = [aws_cloudwatch_log_group.control_runtime.name]
      service_names   = ["${var.agent_name}.${var.control_endpoint_name}"]
    }
  }

  # Every evaluator in the workflow's quality-gates must be here. Each extra
  # evaluator scores every session of both variants, so keep only the gate.
  evaluator {
    evaluator_id = aws_bedrockagentcore_evaluator.support_workflow.evaluator_id
  }

  rule {
    sampling_config {
      sampling_percentage = var.evaluation_sampling_percentage
    }

    session_config {
      session_timeout_minutes = var.evaluation_session_timeout_minutes
    }
  }

  lifecycle {
    precondition {
      condition     = !strcontains(var.agent_name, var.control_endpoint_name)
      error_message = "agent_name must not contain control_endpoint_name: the release gate replaces it with \"treatment\" in the service and log group names."
    }
  }

  depends_on = [aws_iam_role_policy.evaluation]
}
```

With our values, the log group is `/aws/bedrock-agentcore/runtimes/<runtime-id>-control` and the service name is `showcase_agent.control`. The configuration scores 100% of the sessions and treats a session as complete after 2 idle minutes.

For every release, the action makes one copy of this configuration per variant by replacing `control` with `treatment` in the log group and the service name. Both must contain `control`, and the agent name must not.

AgentCore creates the runtime's log group on the first invocation, so Terraform creates it up front. That way, the configuration can point to it from the beginning.

The template contains only the gate evaluator. Every evaluator in the configuration scores every session of both variants during a release, so an LLM-as-a-judge evaluator there adds judge-model cost to each session. AgentCore also locks an evaluator that an enabled online configuration references, so you must disable the configuration before you change the evaluator's definition. Changing only the Lambda code is fine.

## Prerequisites

Before you deploy this solution, set up your environment with the following:

- An AWS account and a Region with AgentCore Runtime, Gateway, Evaluations and A/B testing. We use `eu-central-1`.
- Access in Amazon Bedrock to `openai.gpt-oss-20b-1:0` for the agent and to `openai.gpt-oss-120b-1:0` for the on-demand judge.
- CloudWatch Transaction Search enabled, because AgentCore Evaluations reads the spans from it. `aws xray get-trace-segment-destination` should return `CloudWatchLogs` and `ACTIVE`.
- The GitHub OIDC identity provider in IAM (`token.actions.githubusercontent.com`).
- Terraform v1.14 or later with the AWS provider v6.63 or later.
- Docker with buildx, the AWS Command Line Interface (AWS CLI), the GitHub CLI and [uv](https://docs.astral.sh/uv/) with Python v3.12 or later.
- A GitHub repository. The build runs on `ubuntu-24.04-arm`, which is free for public repositories.

On the AWS side, the action expects an ARM64 image in Amazon Elastic Container Registry (Amazon ECR), a runtime with a ready `control` endpoint, a dedicated HTTP gateway, one enabled online evaluation configuration, an A/B test role and a deploy role. The Terraform in the repository creates all of them, plus the on-demand judge.

## The infrastructure

The `infra/` folder contains one file per concern:

| File | Resources |
|---|---|
| `ecr.tf` | The image repository |
| `runtime.tf` | The runtime and the `control` endpoint |
| `gateway.tf` | The HTTP gateway and its `control` target |
| `evaluations.tf` | The evaluator Lambda function, the evaluator and the online evaluation configuration |
| `evaluators.tf` | The LLM-as-a-judge evaluator for on-demand evaluation |
| `iam.tf` | Roles for the runtime, the gateway and the evaluations |
| `release.tf` | The A/B test role and the GitHub deploy role |

A few details make Terraform and the action work together.

The action deploys releases by changing the runtime's image and the `control` endpoint's version, so Terraform creates both resources and ignores those two attributes:

```hcl
resource "aws_bedrockagentcore_agent_runtime" "agent" {
  agent_runtime_name = var.agent_name
  description        = "Showcase agent released through the AgentCore A/B release gate"
  role_arn           = aws_iam_role.runtime.arn

  agent_runtime_artifact {
    container_configuration {
      container_uri = "${aws_ecr_repository.agent.repository_url}:${var.initial_image_tag}"
    }
  }

  network_configuration {
    network_mode = "PUBLIC"
  }

  protocol_configuration {
    server_protocol = "HTTP"
  }

  # The release gate action deploys new images with UpdateAgentRuntime.
  # Terraform only creates the Runtime and must not roll those releases back.
  lifecycle {
    ignore_changes = [agent_runtime_artifact]
  }

  depends_on = [aws_iam_role_policy.runtime]
}

# Stable endpoint the action treats as the control arm. The action also
# creates a "treatment" endpoint itself on its first run.
resource "aws_bedrockagentcore_agent_runtime_endpoint" "control" {
  name                  = var.control_endpoint_name
  agent_runtime_id      = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_id
  agent_runtime_version = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_version
  description           = "Control endpoint serving the approved agent version"

  # The action repoints this endpoint on promotion and rollback.
  lifecycle {
    ignore_changes = [agent_runtime_version]
  }
}
```

The gateway leaves `protocol_type` unset, which creates an HTTP gateway, the type that accepts runtime HTTP targets. The target's `qualifier` is the endpoint name:

```hcl
# No protocol_type: an HTTP Gateway that routes traffic straight to
# AgentCore Runtime targets, which is what the A/B test splits.
resource "aws_bedrockagentcore_gateway" "agent" {
  name            = "${var.project_name}-gateway"
  description     = "Dedicated HTTP gateway fronting the agent for A/B releases"
  role_arn        = aws_iam_role.gateway.arn
  authorizer_type = "AWS_IAM"
}

# Control target. The action looks it up by name (control-endpoint-name)
# and adds its own treatment target next to it during a release.
resource "aws_bedrockagentcore_gateway_target" "control" {
  name               = var.control_endpoint_name
  gateway_identifier = aws_bedrockagentcore_gateway.agent.gateway_id
  description        = "Routes to the agent's control endpoint"

  credential_provider_configuration {
    gateway_iam_role {}
  }

  target_configuration {
    http {
      agentcore_runtime {
        arn       = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn
        qualifier = aws_bedrockagentcore_agent_runtime_endpoint.control.name
      }
    }
  }
}
```

AgentCore assumes the A/B test role to manage the gateway rules and to read the evaluation data. Its policy follows the AWS documentation on A/B testing prerequisites.

GitHub Actions assumes the deploy role. It has the permissions listed in the action's README, plus ECR push for the build and `bedrock-agentcore:InvokeGateway` for the traffic job. Its `max_session_duration` is 4 hours, because the action uses the credentials for the whole test.

## Deploy the solution

The infrastructure is deployed by hand with Terraform. Only agent releases run in GitHub Actions.

The runtime needs an image when it is created, so the ECR repository comes first. Build the bootstrap image from version 1 of the prompt (in the repository's README), not from the repository's version 3. Complete the following steps:

1. Create the ECR repository, push version 1 of the agent with the tag `bootstrap`, and create the rest of the infrastructure:

```bash
terraform -chdir=infra init
terraform -chdir=infra apply -target=aws_ecr_repository.agent

REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
aws ecr get-login-password --region eu-central-1 \
  | docker login --username AWS --password-stdin "${REPO%%/*}"

# Version 1 with the tag "bootstrap". ARM64, built from the repository root.
docker buildx build --platform linux/arm64 --provenance=false \
  -f src/agent/Dockerfile -t "$REPO:bootstrap" --push .

terraform -chdir=infra apply
```

```text
Outputs:
ab_test_role_arn              = "arn:aws:iam::123456789012:role/agentcore-release-showcase-ab-test"
aws_region                    = "eu-central-1"
control_endpoint_name         = "control"
ecr_repository_url            = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/agentcore-release-showcase"
evaluation_config_id          = "showcase_agent_control_eval-TIuENTGKTP"
gateway_id                    = "agentcore-release-showcase-gateway-tbxw0u5whz"
github_deploy_role_arn        = "arn:aws:iam::123456789012:role/agentcore-release-showcase-github-deploy"
order_grounding_evaluator_id  = "showcase_agent_order_grounding-EiF5HKA6EO"
runtime_id                    = "showcase_agent-5d8CQj7PYA"
support_workflow_evaluator_id = "showcase_agent_support_workflow-6D9E4mCaGA"
```

2. Generate traffic against version 1. `scripts/traffic.sh` sends customer questions to the gateway, one new session per request, signed with SigV4. Each request is the following `curl` call, where `$url` is `https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/control/invocations`:

```bash
status=$(curl -sS -o /tmp/traffic-response.json -w '%{http_code}' --max-time 120 \
  --aws-sigv4 "aws:amz:${region}:bedrock-agentcore" \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
  ${AWS_SESSION_TOKEN:+-H "x-amz-security-token: ${AWS_SESSION_TOKEN}"} \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: ${session_id}" \
  -d "$body" -X POST "$url") || status=000
```

The script takes the gateway ID, the Region, the duration, the pause between requests and the number of parallel customers. The following output shows 40 seconds of traffic against version 1 with two customers (the script prints only the beginning of each answer):

```text
$ scripts/traffic.sh agentcore-release-showcase-gateway-tbxw0u5whz eu-central-1 40 5 2
[w1 1] Can I return the headphones from order ORD-1001? -> Sure! The HD‑200 Wireless Headphones in order **ORD‑1001** are still return‑eligible. You can return them anytime up to **10 Oct 2026** for a full ref
[w0 1] Hi, I'm ana@example.com. Where is my keyboard? -> Your mechanical keyboard (order ORD‑1002) is on its way!   - **Tracking number:** JD014600003456789012 (DHL)   - **Estimated delivery:** 29 September 20
[w1 2] I'm ben@example.com, I want to return my monitor. -> I’m sorry, but the 30‑day return window for the 27‑inch Monitor (SKU: MN‑270) closed on **2026‑09‑02**.    Because that date has passed, you’re
[w0 2] Please open a return for the USB-C cable in ORD-1001, it doesn't charge my phone. -> Great news! Your return has been processed.  * **Return ID**: **RMA‑5001**   * **Item**: USB‑C Cable (SKU CB‑010)   * **Reason**:
...
Worker 1 sent 4 requests, 0 failed.
Worker 0 sent 5 requests, 0 failed.
```

Version 1 reads the data: the DHL tracking number, the return window of the monitor and the RMA number all come from the tools. The first answer also shows why the release gate needs more than rules on identifiers: the return deadline of the headphones is 2026-10-14, not 10 October. That kind of error is what the on-demand judge at the end of this post is for.

3. Wait a few minutes for online evaluation to score the sessions. Every session is scored by the template configuration, so you can check the baseline before the first release. Version 1 follows the workflow in most turns and fails the return requests where it asks for a SKU; the control column of the first release below shows its scores on 191 turns.

## Configure the release workflow

`.github/workflows/release.yml` runs on every push to `main` that changes the agent. It has three jobs:

- `publish` builds the ARM64 image on an ARM runner and pushes it to Amazon ECR with the commit SHA as the tag (about 1 minute).
- `deploy` runs the release gate (about 20 minutes).
- `traffic` runs in parallel with `deploy` and plays the customers (about 25 minutes). The action only observes traffic; in production, your users generate it.

The following step calls the action:

{% raw %}
```yaml
- uses: alvarog2491/agentcore-ab-release-gate@c3b197619c73e45d395bb3eb9a535ddc464ed254 # v1.1.2
  with:
    step: auto
    image-uri: ${{ steps.image.outputs.uri }}
    runtime-id: ${{ vars.AGENTCORE_RUNTIME_ID }}
    gateway-id: ${{ vars.AGENTCORE_GATEWAY_ID }}
    aws-region: ${{ env.AWS_REGION }}
    evaluation-config-id: ${{ vars.AGENTCORE_EVALUATION_CONFIG_ID }}
    ab-test-role-arn: ${{ vars.AGENTCORE_AB_TEST_ROLE_ARN }}
    # Showcase-sized test: 15 min observation, even split so the treatment
    # collects enough sessions quickly. Production default is 7200 / 80-20.
    duration-seconds: ${{ env.OBSERVATION_SECONDS }}
    control-weight: '50'
    treatment-weight: '50'
    # Minimum treatment score per evaluator ID (`terraform output support_workflow_evaluator_id`).
    # support_workflow scores each turn 1 if it follows the store's support
    # workflow (looks facts up, invents no IDs, never asks for a SKU, checks
    # eligibility before create_return), else 0: >= 90% of treatment turns.
    quality-gates: >-
      {"showcase_agent_support_workflow-6D9E4mCaGA": 0.9}
```
{% endraw %}

`step: auto` observes and promotes in the same job. We changed the following inputs from their defaults:

- `duration-seconds` is 900 (default 7200), a 15-minute observation sized for this showcase.
- The split is 50/50 (default 80/20), so the treatment collects enough sessions quickly.
- `scoring-lag-seconds` is 300 (default 120). Online evaluation scores sessions in batches a few minutes after they end, so the action waits until the number of scored sessions has been stable for 5 minutes before it decides.
- `quality-gates` requires a score of at least 0.9, which means that at least 90% of the treatment turns follow every workflow rule.

The remaining inputs keep their defaults. `require-significance` is `true`, so the p-value must be below 0.05, and `evaluation-timeout-seconds` is 1800.

The keys of `quality-gates` are full evaluator IDs, including the random suffix that AWS adds to custom evaluators. Copy the ID from `terraform output support_workflow_evaluator_id` after each apply.

The traffic job runs four parallel customers that send the same 12 questions in a loop, one new session per question, so each variant collects enough sessions for a significant result in 15 minutes. Seven of the questions are return requests or return questions, because most of the workflow rules apply to returns. In production, your users generate the traffic, and the mix is whatever they ask.

{% details The full release.yml %}
{% raw %}
```yaml
name: Release agent

# Builds the agent image, publishes it to ECR, then releases it through the
# AgentCore A/B Release Gate: the new image runs as a treatment next to the
# current control version and is promoted only if every quality gate passes.

on:
  push:
    branches: [main]
    paths:
      - src/agent/**
      - pyproject.toml
      - uv.lock
      - .github/workflows/release.yml
  workflow_dispatch:

permissions:
  contents: read
  id-token: write # OIDC to assume the AWS deploy role

env:
  AWS_REGION: ${{ vars.AWS_REGION }}
  ECR_REPOSITORY: agentcore-release-showcase # infra/ecr.tf (var.project_name)
  OBSERVATION_SECONDS: 900

jobs:
  publish:
    name: Publish image to ECR
    # AgentCore Runtime needs linux/arm64; build natively instead of emulating.
    runs-on: ubuntu-24.04-arm
    timeout-minutes: 30
    outputs:
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1

      - uses: aws-actions/configure-aws-credentials@e1253824e5c10ff9df46874f81ed3ec929e19cfd # v6.3.0
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - id: ecr
        uses: aws-actions/amazon-ecr-login@03f1aad4c6c7ffd436567f42f9384779290529bd # v2.1.7

      - uses: docker/setup-buildx-action@f87e5991a6d7451dcb8d9637bfbc97413f497069 # v4.4.1

      - id: build
        uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc # v7.4.0
        with:
          # Repo root is the context: the image needs the uv workspace lockfile.
          context: .
          file: src/agent/Dockerfile
          platforms: linux/arm64
          push: true
          provenance: false # push a single ARM64 manifest, not an index with attestations
          tags: ${{ steps.ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    name: A/B release gate
    needs: publish
    runs-on: ubuntu-latest
    timeout-minutes: 180
    environment: production
    # One release at a time per Gateway; never cancel a running A/B test.
    concurrency:
      group: agentcore-production
      cancel-in-progress: false
    steps:
      - uses: aws-actions/configure-aws-credentials@e1253824e5c10ff9df46874f81ed3ec929e19cfd # v6.3.0
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
          # Must outlast observation + evaluation; the role's MaxSessionDuration must allow it.
          role-duration-seconds: 14400

      # Only the digest crosses jobs: GitHub drops job outputs containing masked
      # values, which the registry URI (AWS account ID) can be.
      - id: image
        run: |
          account_id=$(aws sts get-caller-identity --query Account --output text)
          echo "uri=${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}@${{ needs.publish.outputs.digest }}" >> "$GITHUB_OUTPUT"

      - uses: alvarog2491/agentcore-ab-release-gate@c3b197619c73e45d395bb3eb9a535ddc464ed254 # v1.1.2
        with:
          step: auto
          image-uri: ${{ steps.image.outputs.uri }}
          runtime-id: ${{ vars.AGENTCORE_RUNTIME_ID }}
          gateway-id: ${{ vars.AGENTCORE_GATEWAY_ID }}
          aws-region: ${{ env.AWS_REGION }}
          evaluation-config-id: ${{ vars.AGENTCORE_EVALUATION_CONFIG_ID }}
          ab-test-role-arn: ${{ vars.AGENTCORE_AB_TEST_ROLE_ARN }}
          # Showcase-sized test: 15 min observation, even split so the treatment
          # collects enough sessions quickly. Production default is 7200 / 80-20.
          duration-seconds: ${{ env.OBSERVATION_SECONDS }}
          control-weight: '50'
          treatment-weight: '50'
          # Scores arrive in batches minutes after a session ends; wait for the
          # sample count to stay stable for 5 min so most sessions are counted.
          scoring-lag-seconds: '300'
          # Minimum treatment score per evaluator ID (`terraform output support_workflow_evaluator_id`).
          # support_workflow scores each turn 1 if it follows the store's support
          # workflow (looks facts up, invents no IDs, never asks for a SKU, checks
          # eligibility before create_return), else 0: >= 90% of treatment turns.
          quality-gates: >-
            {"showcase_agent_support_workflow-6D9E4mCaGA": 0.9}

  # The release gate observes real traffic but does not create it. This job
  # plays customers: it sends questions through the Gateway's control target
  # while the deploy job prepares the candidate and runs the A/B test.
  traffic:
    name: Generate traffic
    needs: publish
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1

      - uses: aws-actions/configure-aws-credentials@e1253824e5c10ff9df46874f81ed3ec929e19cfd # v6.3.0
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
          role-duration-seconds: 3600

      # Observation window plus ~10 min for the gate to deploy the candidate
      # and create the treatment endpoint before the A/B test starts. Four
      # parallel customers give each variant enough sessions for significance.
      - run: scripts/traffic.sh "${{ vars.AGENTCORE_GATEWAY_ID }}" "$AWS_REGION" $((OBSERVATION_SECONDS + 600)) 5 4
```
{% endraw %}
{% enddetails %}

## Release version 2: rolled back

With version 2 in `DEFAULT_SYSTEM_PROMPT`, a push to `main` starts the release. The control serves version 1.

The runtime numbers its versions on every image update, and they don't match the agent versions of this post. In our account, runtime version 1 was an earlier bootstrap image, version 1 of the agent ran as runtime version 2, and the release created runtime version 3 for agent version 2.

The following table shows how the run progressed (times in UTC):

| Time | Event |
|---|---|
| 08:30:29 | `publish` starts. Build and push take 81 seconds |
| 08:32:03 | The action resolves the image digest |
| 08:32:05 | The `treatment` endpoint and gateway target from an earlier release are reused |
| 08:32:06 | New image deployed as runtime version 3 |
| 08:32:27 | `treatment` moves to runtime version 3, `control` stays on runtime version 2 |
| 08:32:50 | One evaluation configuration per variant is ready |
| 08:33:02 | The A/B test is running, 50/50 |
| 08:48:02 | End of the 900 seconds of observation |
| 08:49:02 | First results: 27 treatment and 18 control sessions scored |
| 09:18:02 | Final results: 208 treatment and 172 control sessions; the gate fails |
| 09:18:55 | Candidate rolled back |

The following are the main lines of the action's log:

```text
08:32:03 {"event": "candidate-image-resolved", "image": "123456789012.dkr.ecr.eu-central-1.amazonaws.com/agentcore-release-showcase@sha256:4ecb76f76daf15dbb36de65687420bb14743a35bc83d43009995932da32cea0e", "observationSeconds": 900}
08:32:05 {"event": "deployment-prepared", "baseline": "2", "runtime": "showcase_agent-5d8CQj7PYA", "gateway": "agentcore-release-showcase-gateway-tbxw0u5whz"}
08:32:06 {"event": "candidate-runtime-created", "version": "3"}
08:32:27 {"event": "treatment-endpoint-serving", "endpoint": "treatment", "version": "3"}
08:32:28 {"event": "evaluation-config-created", "variant": "control", "evaluationConfigId": "showcase_agent_control_eval_c_aa7922b6-STy4x45q5z"}
08:32:39 {"event": "evaluation-config-created", "variant": "treatment", "evaluationConfigId": "showcase_agent_control_eval_t_13afb08c-3AwTuh8Kay"}
08:32:51 {"event": "ab-test-created", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:33:02 {"event": "ab-test-running", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:48:02 {"event": "evaluation-results-waiting", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:49:02 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 60, "remainingSeconds": 1739, "totalSamplesScored": 27, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
…  (a 'stabilizing-results' line every 30 s while scores arrive)
09:18:02 {"event": "ab-test-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "results": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "isSignificant": true, "absoluteChange": -0.2753801431127012, "percentChange": -29.789550072568936, "pValue": 6.291833366249719e-09, "treatmentSampleSize": 208, "controlSampleSize": 172}}}
09:18:55 Rolled back: control retains version 2
```

{% details The full log of the action (120 events) %}
```text
08:32:03 {"event": "candidate-image-resolved", "image": "123456789012.dkr.ecr.eu-central-1.amazonaws.com/agentcore-release-showcase@sha256:4ecb76f76daf15dbb36de65687420bb14743a35bc83d43009995932da32cea0e", "observationSeconds": 900}
08:32:03 {"event": "deployment-preparing", "controlEndpoint": "control"}
08:32:04 {"event": "treatment-endpoint-existing", "endpoint": "treatment"}
08:32:05 {"event": "treatment-endpoint-ready", "endpoint": "treatment"}
08:32:05 {"event": "gateway-target-existing", "target": "control"}
08:32:05 {"event": "gateway-target-ready", "target": "control", "targetId": "1FXXBFSC0Y"}
08:32:05 {"event": "gateway-target-existing", "target": "treatment"}
08:32:05 {"event": "gateway-target-ready", "target": "treatment", "targetId": "RULIRM0A1H"}
08:32:05 {"event": "deployment-prepared", "baseline": "2", "runtime": "showcase_agent-5d8CQj7PYA", "gateway": "agentcore-release-showcase-gateway-tbxw0u5whz"}
08:32:05 {"event": "candidate-runtime-creating", "image": "123456789012.dkr.ecr.eu-central-1.amazonaws.com/agentcore-release-showcase@sha256:4ecb76f76daf15dbb36de65687420bb14743a35bc83d43009995932da32cea0e"}
08:32:06 {"event": "candidate-runtime-created", "version": "3"}
08:32:16 {"event": "candidate-runtime-ready", "version": "3"}
08:32:16 {"event": "treatment-endpoint-updating", "endpoint": "treatment", "version": "3"}
08:32:27 {"event": "treatment-endpoint-serving", "endpoint": "treatment", "version": "3"}
08:32:27 {"event": "evaluation-config-template-loading", "evaluationConfigId": "showcase_agent_control_eval-TIuENTGKTP"}
08:32:27 {"event": "evaluation-config-creating", "variant": "control"}
08:32:28 {"event": "evaluation-config-created", "variant": "control", "evaluationConfigId": "showcase_agent_control_eval_c_aa7922b6-STy4x45q5z"}
08:32:28 {"event": "evaluation-config-waiting", "variant": "control", "evaluationConfigId": "showcase_agent_control_eval_c_aa7922b6-STy4x45q5z", "status": "CREATING"}
08:32:38 {"event": "evaluation-config-ready", "variant": "control", "evaluationConfigId": "showcase_agent_control_eval_c_aa7922b6-STy4x45q5z"}
08:32:38 {"event": "evaluation-config-creating", "variant": "treatment"}
08:32:39 {"event": "evaluation-config-created", "variant": "treatment", "evaluationConfigId": "showcase_agent_control_eval_t_13afb08c-3AwTuh8Kay"}
08:32:39 {"event": "evaluation-config-waiting", "variant": "treatment", "evaluationConfigId": "showcase_agent_control_eval_t_13afb08c-3AwTuh8Kay", "status": "CREATING"}
08:32:50 {"event": "evaluation-config-ready", "variant": "treatment", "evaluationConfigId": "showcase_agent_control_eval_t_13afb08c-3AwTuh8Kay"}
08:32:50 {"event": "ab-test-creating", "controlWeight": 50, "treatmentWeight": 50, "gatewayArn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:gateway/agentcore-release-showcase-gateway-tbxw0u5whz"}
08:32:51 {"event": "ab-test-created", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:33:02 {"event": "ab-test-running", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:33:02 {"event": "listening-for-connections", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "observationSeconds": 900, "message": "A/B test is running; waiting for Gateway connections and evaluator results."}
08:33:32 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 30, "remainingSeconds": 870, "abTestStatus": "RUNNING"}
08:34:02 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 60, "remainingSeconds": 840, "abTestStatus": "RUNNING"}
08:34:32 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 90, "remainingSeconds": 810, "abTestStatus": "RUNNING"}
08:35:02 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 120, "remainingSeconds": 780, "abTestStatus": "RUNNING"}
08:35:32 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 150, "remainingSeconds": 750, "abTestStatus": "RUNNING"}
08:36:03 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 180, "remainingSeconds": 720, "abTestStatus": "RUNNING"}
08:36:33 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 210, "remainingSeconds": 690, "abTestStatus": "RUNNING"}
08:37:03 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 241, "remainingSeconds": 659, "abTestStatus": "RUNNING"}
08:37:33 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 271, "remainingSeconds": 629, "abTestStatus": "RUNNING"}
08:38:03 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 301, "remainingSeconds": 599, "abTestStatus": "RUNNING"}
08:38:34 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 331, "remainingSeconds": 569, "abTestStatus": "RUNNING"}
08:39:04 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 362, "remainingSeconds": 538, "abTestStatus": "RUNNING"}
08:39:34 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 392, "remainingSeconds": 508, "abTestStatus": "RUNNING"}
08:40:04 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 422, "remainingSeconds": 478, "abTestStatus": "RUNNING"}
08:40:34 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 452, "remainingSeconds": 448, "abTestStatus": "RUNNING"}
08:41:05 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 482, "remainingSeconds": 418, "abTestStatus": "RUNNING"}
08:41:35 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 513, "remainingSeconds": 387, "abTestStatus": "RUNNING"}
08:42:05 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 543, "remainingSeconds": 357, "abTestStatus": "RUNNING"}
08:42:35 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 573, "remainingSeconds": 327, "abTestStatus": "RUNNING"}
08:43:05 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 603, "remainingSeconds": 297, "abTestStatus": "RUNNING"}
08:43:36 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 633, "remainingSeconds": 267, "abTestStatus": "RUNNING"}
08:44:06 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 663, "remainingSeconds": 237, "abTestStatus": "RUNNING"}
08:44:36 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 694, "remainingSeconds": 206, "abTestStatus": "RUNNING"}
08:45:06 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 724, "remainingSeconds": 176, "abTestStatus": "RUNNING"}
08:45:36 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 754, "remainingSeconds": 146, "abTestStatus": "RUNNING"}
08:46:06 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 784, "remainingSeconds": 116, "abTestStatus": "RUNNING"}
08:46:37 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 814, "remainingSeconds": 86, "abTestStatus": "RUNNING"}
08:47:07 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 845, "remainingSeconds": 55, "abTestStatus": "RUNNING"}
08:47:37 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 875, "remainingSeconds": 25, "abTestStatus": "RUNNING"}
08:48:02 {"event": "observing", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 900, "remainingSeconds": 0, "abTestStatus": "RUNNING"}
08:48:02 {"event": "evaluation-results-waiting", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553"}
08:48:02 {"event": "waiting-for-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 0, "remainingSeconds": 1799, "totalSamplesScored": 0, "evaluatorsReady": [], "evaluatorsWaiting": ["showcase_agent_support_workflow-6D9E4mCaGA"], "partialResults": {}}
08:48:32 {"event": "waiting-for-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 30, "remainingSeconds": 1769, "totalSamplesScored": 0, "evaluatorsReady": [], "evaluatorsWaiting": ["showcase_agent_support_workflow-6D9E4mCaGA"], "partialResults": {}}
08:49:02 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 60, "remainingSeconds": 1739, "totalSamplesScored": 27, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:49:32 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 90, "remainingSeconds": 1709, "totalSamplesScored": 27, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:50:03 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 120, "remainingSeconds": 1679, "totalSamplesScored": 27, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:50:33 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 150, "remainingSeconds": 1649, "totalSamplesScored": 27, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:51:03 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 181, "remainingSeconds": 1618, "totalSamplesScored": 27, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:51:33 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 211, "remainingSeconds": 1588, "totalSamplesScored": 27, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:52:03 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 241, "remainingSeconds": 1558, "totalSamplesScored": 27, "stableForSeconds": 181, "remainingScoringLagSeconds": 118, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:52:33 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 271, "remainingSeconds": 1528, "totalSamplesScored": 27, "stableForSeconds": 211, "remainingScoringLagSeconds": 88, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:53:04 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 301, "remainingSeconds": 1498, "totalSamplesScored": 27, "stableForSeconds": 241, "remainingScoringLagSeconds": 58, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:53:34 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 331, "remainingSeconds": 1468, "totalSamplesScored": 27, "stableForSeconds": 271, "remainingScoringLagSeconds": 28, "awaitingSignificance": true, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.7037037037037037, "treatmentSamples": 27, "controlSamples": 18, "pValue": 0.34901600461362947}}}
08:54:04 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 362, "remainingSeconds": 1437, "totalSamplesScored": 61, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:54:34 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 392, "remainingSeconds": 1407, "totalSamplesScored": 61, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:55:04 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 422, "remainingSeconds": 1377, "totalSamplesScored": 61, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:55:34 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 452, "remainingSeconds": 1347, "totalSamplesScored": 61, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:56:05 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 482, "remainingSeconds": 1317, "totalSamplesScored": 61, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:56:35 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 512, "remainingSeconds": 1287, "totalSamplesScored": 61, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:57:05 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 543, "remainingSeconds": 1256, "totalSamplesScored": 61, "stableForSeconds": 181, "remainingScoringLagSeconds": 118, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:57:35 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 573, "remainingSeconds": 1226, "totalSamplesScored": 61, "stableForSeconds": 211, "remainingScoringLagSeconds": 88, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:58:05 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 603, "remainingSeconds": 1196, "totalSamplesScored": 61, "stableForSeconds": 241, "remainingScoringLagSeconds": 58, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:58:35 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 633, "remainingSeconds": 1166, "totalSamplesScored": 61, "stableForSeconds": 271, "remainingScoringLagSeconds": 28, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.5737704918032787, "treatmentSamples": 61, "controlSamples": 44, "pValue": 0.0001149945455902997}}}
08:59:06 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 663, "remainingSeconds": 1136, "totalSamplesScored": 99, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
08:59:36 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 694, "remainingSeconds": 1105, "totalSamplesScored": 99, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:00:06 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 724, "remainingSeconds": 1075, "totalSamplesScored": 99, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:00:36 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 754, "remainingSeconds": 1045, "totalSamplesScored": 99, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:01:06 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 784, "remainingSeconds": 1015, "totalSamplesScored": 99, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:01:36 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 814, "remainingSeconds": 985, "totalSamplesScored": 99, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:02:07 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 844, "remainingSeconds": 955, "totalSamplesScored": 99, "stableForSeconds": 181, "remainingScoringLagSeconds": 118, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:02:37 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 875, "remainingSeconds": 924, "totalSamplesScored": 99, "stableForSeconds": 211, "remainingScoringLagSeconds": 88, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:03:07 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 905, "remainingSeconds": 894, "totalSamplesScored": 99, "stableForSeconds": 241, "remainingScoringLagSeconds": 58, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:03:37 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 935, "remainingSeconds": 864, "totalSamplesScored": 99, "stableForSeconds": 271, "remainingScoringLagSeconds": 28, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6464646464646465, "treatmentSamples": 99, "controlSamples": 67, "pValue": 0.00017551537736039635}}}
09:04:07 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 965, "remainingSeconds": 834, "totalSamplesScored": 146, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:04:37 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 995, "remainingSeconds": 804, "totalSamplesScored": 146, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:05:08 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1025, "remainingSeconds": 774, "totalSamplesScored": 146, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:05:38 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1056, "remainingSeconds": 743, "totalSamplesScored": 146, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:06:08 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1086, "remainingSeconds": 713, "totalSamplesScored": 146, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:06:38 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1116, "remainingSeconds": 683, "totalSamplesScored": 146, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:07:08 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1146, "remainingSeconds": 653, "totalSamplesScored": 146, "stableForSeconds": 181, "remainingScoringLagSeconds": 118, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:07:39 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1176, "remainingSeconds": 623, "totalSamplesScored": 146, "stableForSeconds": 211, "remainingScoringLagSeconds": 88, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:08:09 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1207, "remainingSeconds": 592, "totalSamplesScored": 146, "stableForSeconds": 241, "remainingScoringLagSeconds": 58, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:08:39 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1237, "remainingSeconds": 562, "totalSamplesScored": 146, "stableForSeconds": 271, "remainingScoringLagSeconds": 28, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6506849315068494, "treatmentSamples": 146, "controlSamples": 109, "pValue": 1.9215642261708763e-07}}}
09:09:09 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1267, "remainingSeconds": 532, "totalSamplesScored": 202, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:09:39 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1297, "remainingSeconds": 502, "totalSamplesScored": 202, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:10:09 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1327, "remainingSeconds": 472, "totalSamplesScored": 202, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:10:40 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1357, "remainingSeconds": 442, "totalSamplesScored": 202, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:11:10 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1387, "remainingSeconds": 412, "totalSamplesScored": 202, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:11:40 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1418, "remainingSeconds": 381, "totalSamplesScored": 202, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:12:10 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1448, "remainingSeconds": 351, "totalSamplesScored": 202, "stableForSeconds": 180, "remainingScoringLagSeconds": 119, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:12:40 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1478, "remainingSeconds": 321, "totalSamplesScored": 202, "stableForSeconds": 210, "remainingScoringLagSeconds": 89, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:13:10 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1508, "remainingSeconds": 291, "totalSamplesScored": 202, "stableForSeconds": 241, "remainingScoringLagSeconds": 58, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:13:40 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1538, "remainingSeconds": 261, "totalSamplesScored": 202, "stableForSeconds": 271, "remainingScoringLagSeconds": 28, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6534653465346535, "treatmentSamples": 202, "controlSamples": 156, "pValue": 4.629157118792766e-08}}}
09:14:10 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1568, "remainingSeconds": 231, "totalSamplesScored": 208, "stableForSeconds": 0, "remainingScoringLagSeconds": 300, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:14:41 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1598, "remainingSeconds": 201, "totalSamplesScored": 208, "stableForSeconds": 30, "remainingScoringLagSeconds": 269, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:15:11 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1628, "remainingSeconds": 171, "totalSamplesScored": 208, "stableForSeconds": 60, "remainingScoringLagSeconds": 239, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:15:41 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1659, "remainingSeconds": 140, "totalSamplesScored": 208, "stableForSeconds": 90, "remainingScoringLagSeconds": 209, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:16:11 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1689, "remainingSeconds": 110, "totalSamplesScored": 208, "stableForSeconds": 120, "remainingScoringLagSeconds": 179, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:16:41 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1719, "remainingSeconds": 80, "totalSamplesScored": 208, "stableForSeconds": 150, "remainingScoringLagSeconds": 149, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:17:11 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1749, "remainingSeconds": 50, "totalSamplesScored": 208, "stableForSeconds": 180, "remainingScoringLagSeconds": 119, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:17:41 {"event": "stabilizing-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "elapsedSeconds": 1779, "remainingSeconds": 20, "totalSamplesScored": 208, "stableForSeconds": 210, "remainingScoringLagSeconds": 89, "awaitingSignificance": false, "partialResults": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "treatmentSamples": 208, "controlSamples": 172, "pValue": 6.291833366249719e-09}}}
09:18:02 {"event": "ab-test-results", "abTestId": "agentcore_release_gate_d68bfb89-497f74b553", "results": {"showcase_agent_support_workflow-6D9E4mCaGA": {"mean": 0.6490384615384616, "isSignificant": true, "absoluteChange": -0.2753801431127012, "percentChange": -29.789550072568936, "pValue": 6.291833366249719e-09, "treatmentSampleSize": 208, "controlSampleSize": 172}}}
09:18:55 Rolled back: control retains version 2
```
{% enddetails %}

During the test, each endpoint served its own version. The following output lists the endpoints as (name, live version, target version, status) and the gateway targets:

```text
endpoints: [('treatment', '3', None, 'READY'), ('control', '2', None, 'READY'), ('DEFAULT', '3', None, 'READY')]
targets: ['control', 'treatment']
```

`DEFAULT` always follows the latest runtime version, so it was already on the candidate. Route users through the gateway.

The following JSON shows the A/B test created by the action, from `GetABTest` (shortened):

```json
{
  "abTestId": "agentcore_release_gate_d68bfb89-497f74b553",
  "executionStatus": "RUNNING",
  "variants": [
    {"name": "C",  "weight": 50, "variantConfiguration": {"target": {"name": "control"}}},
    {"name": "T1", "weight": 50, "variantConfiguration": {"target": {"name": "treatment"}}}
  ],
  "gatewayFilter": {"targetPaths": ["/control/*"]},
  "evaluationConfig": {
    "perVariantOnlineEvaluationConfig": [
      {"name": "C",  "onlineEvaluationConfigArn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:online-evaluation-config/showcase_agent_control_eval_c_aa7922b6-STy4x45q5z"},
      {"name": "T1", "onlineEvaluationConfigArn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:online-evaluation-config/showcase_agent_control_eval_t_13afb08c-3AwTuh8Kay"}
    ]
  }
}
```

Clients keep calling `/control/invocations`, and the gateway decides for each session which target answers. The traffic job sent 601 requests in 25 minutes. 15 of them failed: 11 with HTTP 424 (the runtime returned a 500 when the model produced an invalid tool name) and 4 with a client timeout.

The following answers to "Please open a return for the USB-C cable in ORD-1001, it doesn't charge my phone." come from the runtime logs of each endpoint (shortened). The control checked eligibility and opened the return. The treatment opened it directly in some sessions and asked the customer for the SKU in others:

```text
control:   ✅ A return has been opened for the USB‑C Cable (SKU: CB‑010) in order ORD‑1001.
           * Return reference: RMA‑5001 * Reason: It doesn't charge your phone. ...
treatment: Return RMA‑5001 created for the CB‑010 USB‑C cable. A prepaid label is sent to
           your email, and the €12.50 refund will be processed once the item arrives ...
treatment: Could you let me know the SKU of the USB‑C cable in order ORD‑1001?
treatment: I'm sorry, but the "USB-C cable" isn't listed in order ORD‑1001. Could you let me
           know the exact SKU of the cable you'd like to return?
```

The last answer is wrong as well as unhelpful: the cable is in the order, and the treatment never read it.

## Interpreting the release results

Every turn is scored on its own. The following are two result records, one from each variant (shortened):

```json
{
  "service.name": "showcase_agent.treatment",
  "attributes": {
    "gen_ai.evaluation.name": "showcase_agent_support_workflow",
    "gen_ai.evaluation.score.value": 0.0,
    "gen_ai.evaluation.score.label": "FAIL",
    "gen_ai.evaluation.explanation": "The agent asked the customer for a SKU instead of reading the order with get_order_details. create_return was called for ORD-1001 USB-C CABLE without check_return_eligibility first.",
    "aws.bedrock_agentcore.evaluation_level": "Trace",
    "aws.bedrock_agentcore.experiment.treatment_name": "T1",
    "aws.bedrock_agentcore.experiment.arn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:ab-test/agentcore_release_gate_d68bfb89-497f74b553"
  }
}
```

```json
{
  "service.name": "showcase_agent.control",
  "attributes": {
    "gen_ai.evaluation.name": "showcase_agent_support_workflow",
    "gen_ai.evaluation.score.value": 1.0,
    "gen_ai.evaluation.score.label": "PASS",
    "gen_ai.evaluation.explanation": "The turn follows the support workflow (tool calls: track_shipment).",
    "aws.bedrock_agentcore.evaluation_level": "Trace",
    "aws.bedrock_agentcore.experiment.treatment_name": "C",
    "aws.bedrock_agentcore.experiment.arn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:ab-test/agentcore_release_gate_d68bfb89-497f74b553"
  }
}
```

The `experiment.treatment_name` field (`C` or `T1`) tells AgentCore which variant a score belongs to. The AgentCore SDK (v1.8 or later) adds it to the spans from headers that the gateway sends, so the agent code needs no changes.

The following table counts the scored turns by rule, from the evaluation result log groups of both variants. It includes 19 control turns that were scored after the action's decision. A turn can break more than one rule:

| | Control (version 1) | Treatment (version 2) |
|---|---|---|
| Turns that passed | 176 | 135 |
| Turns that failed | 15 | 73 |
| Asked the customer for a SKU | 14 | 44 |
| `create_return` without `check_return_eligibility` | 0 | 32 |
| Invented identifier | 1 | 0 |

The evaluation results also contain 202 turns without a score. In 197 of them, AgentCore Evaluations couldn't invoke the evaluator Lambda function (`ThrottlingException: Rate Exceeded`), and in 5 the trace had no agent turn. The account's Lambda concurrency quota was the default of 10 for new accounts, and online evaluation invokes the function for a whole batch of sessions at once. Throttled turns are not retried and not counted, so they reduce the sample size but don't bias the result.

The following result is the one the action used for its decision:

```json
"evaluatorMetrics": [{
  "evaluatorArn": "arn:aws:bedrock-agentcore:eu-central-1:123456789012:evaluator/showcase_agent_support_workflow-6D9E4mCaGA",
  "controlStats": {"variantName": "C", "sampleSize": 172, "mean": 0.924},
  "variantResults": [{
    "variantName": "T1",
    "sampleSize": 208,
    "mean": 0.649,
    "absoluteChange": -0.275,
    "percentChange": -29.8,
    "pValue": 6.29e-09,
    "confidenceInterval": {"lower": -0.352, "upper": -0.199},
    "isSignificant": true
  }]
}]
```

Treatment scored 0.649, below the threshold of 0.9, and 0.275 lower than the control's 0.924. The difference is significant (p-value 6.3e-9), so the result isn't noise: version 2 is worse. Two gate conditions failed, the minimum score and the no-regression check, and the action rolled the candidate back. The numbers match the local comparison (0.88 and 0.62) closely.

The action stopped the A/B test, moved `treatment` back to the control's runtime version, and deleted the temporary evaluation configurations. Users stayed on version 1, except for the sessions that the A/B test routed to the treatment while it ran.

```text
endpoints: [('treatment', '2', None, 'READY'), ('control', '2', None, 'READY'), ('DEFAULT', '3', None, 'READY')]
ab tests: [('agentcore_release_gate_d68bfb89-497f74b553', 'ACTIVE', 'STOPPED'), ...]
eval configs left: ['showcase_agent_control_eval-TIuENTGKTP']
```

The release workflow fails when the gate fails, so the commit that introduced version 2 shows a failed check on GitHub. The A/B test stays in AgentCore for later inspection.

The decision came at 09:18:02, exactly 30 minutes after the observation ended, which is the default `evaluation-timeout-seconds`. Online evaluation scored new sessions in batches about every 5 minutes, so the sample count never stayed unchanged for the full 300 seconds of `scoring-lag-seconds`. When the timeout expires and every evaluator has results, the action decides on the latest results instead of failing, so the decision used all 380 scored sessions.

## Release version 3: promoted

Version 3 replaces version 2 in the repository, so the next push undoes the latency change and adds the SKU fix in one release. The control still serves version 1.

<!-- TODO: fill from the release run of version 3. -->

| Time | Event |
|---|---|
| TODO | `publish` starts |
| TODO | The A/B test is running, 50/50 |
| TODO | End of the 900 seconds of observation |
| TODO | Results stable, quality gate passed |
| TODO | Version 3 promoted |

The following are the main lines of the action's log:

```text
TODO
```

The following result is the one the action used for its decision:

```json
TODO
```

<!-- TODO: treatment mean vs threshold, control mean, p-value. -->

After the release, both endpoints serve version 3:

```text
TODO
```

The A/B test is stopped and remains in AgentCore for later inspection. The two temporary evaluation configurations are deleted, and only the template remains.

If you want a person to approve the promotion, split the job into `step: observe`, `step: promote` and `step: rollback`, and run the promote job in a GitHub environment with required reviewers. The action's README has an example.

## Going further: facts that rules can't check

The four rules catch invented identifiers, but they can't tell whether a date, a price or a weekday in the answer is right. For those facts, the repository includes one LLM-as-a-judge evaluator for on-demand evaluation, `showcase_agent_order_grounding` in `infra/evaluators.tf`. It asks a judge model (`openai.gpt-oss-120b-1:0`, larger than the agent's model, at temperature 0) whether every order fact in the answer comes from a tool result, on a Yes (1.0), Partially (0.5) and No (0.0) scale. Its instructions include the store's policy and the fixed "today", so the judge can check dates without guessing.

`scripts/test_agent.py` invokes the deployed agent with three multi-turn sessions (orders, shipping and returns, five turns each) and prints their session IDs. `scripts/evaluate.py` downloads a session's spans from CloudWatch and calls the `Evaluate` API for each evaluator you pass. Wait 3 to 5 minutes after a session ends, because the spans take that long to reach CloudWatch:

```bash
uv run scripts/test_agent.py --runtime-arn "$(terraform -chdir=infra output -raw runtime_arn)" --category shipping

uv run scripts/evaluate.py --runtime-id "$(terraform -chdir=infra output -raw runtime_id)" \
  --session-id "<session-id>" \
  --evaluators "showcase_agent_order_grounding,Builtin.Faithfulness,Builtin.GoalSuccessRate"
```

The script resolves custom evaluator names to their IDs, sends the trace IDs of every turn for TRACE evaluators (in batches of 10, the API limit), and saves a Markdown report under `results/` next to the downloaded spans.

We ran it on a shipping session of the agent, on an earlier deployment with different resource IDs:

| Evaluator | Level | Scored | Mean score |
|---|---|---|---|
| `showcase_agent_order_grounding` | TRACE | 5 | 0.70 |
| `Builtin.Faithfulness` | TRACE | 5 | 0.90 |
| `Builtin.GoalSuccessRate` | SESSION | 1 | 1.00 |

The session achieved its goals, and the tracking number, carrier and delivery estimate were all correct. The last answer said that ORD-1001 was delivered on "Friday, 14 September 2026". The date comes from the tool result. The weekday is invented: 2026-09-14 is a Monday, and no tool returned a weekday. The custom grounding evaluator flagged it:

```text
Partially (0.5): ... The day-of-week 'Friday' is not present in any tool result — it is derived
by the agent independently. However, 14 September 2026 does indeed fall on a Monday, not a
Friday, making this an invented and incorrect fact not grounded in any tool result.
```

The grounding evaluator also lowered the first turn, where the agent suggested that ORD-1002 was "usually the most recent one" although ORD-1004 is newer, and the second turn, where it assumed the keyboard was in ORD-1002 without reading the order. `Builtin.Faithfulness` rated the weekday turn Generally Yes (0.75). The custom evaluator lists exactly which facts must come from tools, so it scores invented details more strictly.

The release gate's rules passed every turn of that session: every identifier in the answers came from a tool. The judge found what the rules can't express. Once you trust a judge like this one, you can add it to the online evaluation configuration and to `quality-gates`, and accept its cost on every scored session.

## Lessons learned

The showcase surfaced the following lessons:

- **Encode the workflow, not the change.** The evaluator's rules come from how the store wants support to work. They caught a regression that nobody wrote them for, and they keep working for future releases.
- **Measure a candidate locally before you release it.** Running both prompts against the same questions with the evaluator's rules showed, in minutes, whether the A/B test could detect the difference. A difference smaller than what the observation window can measure leads to a rollback for lack of significance, not for quality.
- **With significance required, only improvements get promoted.** A candidate as good as the control is rolled back. Turn off `require-significance` if you want to release changes that only have to not be worse.
- **Send traffic through the gateway during the test.** The action scores only the sessions that go through the gateway while the test runs, and it promotes only with scored sessions.
- **Scores arrive late, in batches.** A session is scored after it has been idle for the configured timeout (2 minutes here), plus processing time, and online evaluation delivers the scores in batches about every 5 minutes. With `scoring-lag-seconds` at 300, the count never stayed stable long enough, and the action decided at its 30-minute evaluation timeout with all the scored sessions. For a real release, keep the default observation of 2 hours.
- **Raise the Lambda concurrency quota for code-based evaluators.** Online evaluation invokes the evaluator for a whole batch of sessions at once. With the default quota of 10 concurrent executions for new accounts, about a third of the evaluations were throttled and never scored. Request a higher quota in Service Quotas (Lambda, Concurrent executions) before you rely on a code-based gate.
- **Quality gates need full evaluator IDs.** The suffix changes if you recreate the evaluator.
- **Names around the control endpoint must contain `control`**, because the action builds the treatment names by replacing it with `treatment`.
- **Terraform must ignore the runtime image and the endpoint version**, or it undoes the releases.
- **Size the traffic for the difference you expect.** Version 3 is better than version 1 by about 0.1 on a 0-to-1 score. With one customer sending a question every 10 seconds, each variant gets only a few scored sessions in 15 minutes, too few for a significant result. With four customers in parallel, the rollback decision used 380 scored sessions.
- **Failed requests are invisible to the evaluator.** Requests fail when the model returns an invalid tool name. They produce no answer to score, so a version that fails more often isn't penalized by a quality evaluator. Watch the error rate separately.

For production, we recommend the following:

- Keep the defaults: `duration-seconds: 7200` and an 80/20 split.
- Use real user traffic.
- Add more evaluators to the template configuration and to `quality-gates`.
- Pin the action to a commit SHA, as in the workflow above.
- Keep the `concurrency` group, so two releases never use the same gateway at the same time.
- Make sure that `role-duration-seconds` and the role's `MaxSessionDuration` cover the whole test.
- If the workflow runs on pull requests, pass `github-token` (with `pull-requests: write`) so the action comments the result on the pull request.

## Clean up

To avoid recurring charges, clean up your AWS account after trying the solution. The action creates the `treatment` endpoint, the `treatment` gateway target and the A/B tests itself, so Terraform doesn't manage them. Delete the stopped A/B tests of the gateway with the `DeleteABTest` API (for example, `boto3.client("bedrock-agentcore").delete_ab_test(abTestId=...)`), delete the `treatment` target and endpoint, and then run `terraform destroy`:

```bash
aws bedrock-agentcore-control list-gateway-targets --gateway-identifier <gateway-id>
aws bedrock-agentcore-control delete-gateway-target --gateway-identifier <gateway-id> --target-id <treatment-target-id>
aws bedrock-agentcore-control delete-agent-runtime-endpoint --agent-runtime-id <runtime-id> --endpoint-name treatment
terraform -chdir=infra destroy
```

AgentCore creates some log groups itself, so Terraform doesn't delete them: the online evaluation results (`/aws/bedrock-agentcore/evaluations/results/<config-id>`, one per configuration, including the per-variant copies of an A/B release) and the log group of the runtime's `DEFAULT` endpoint. List and delete them:

```bash
aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/evaluations/results/showcase_agent \
  --query 'logGroups[].logGroupName' --output text
aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/showcase_agent \
  --query 'logGroups[].logGroupName' --output text
aws logs delete-log-group --log-group-name <log-group-name>
```

The local `results/` folder contains the downloaded spans and evaluation reports, and is ignored by Git.

## Conclusion

In this post, we showed how to release a new version of an Amazon Bedrock AgentCore agent only after it outperforms the current version on live traffic, and how the same gate rolls back a change that looks harmless and makes the agent worse.

AgentCore Runtime endpoints, an AgentCore Gateway A/B test and online evaluation work together with the AgentCore A/B Release Gate GitHub Action to deploy a candidate, score both variants, and decide based on statistically significant results. A deterministic, code-based evaluator that encodes the store's support workflow as four rules made the decision: it rolled back a latency optimization that opened returns without checking eligibility first, and it promoted a prompt fix that stopped the agent from asking customers for SKUs. For the facts that rules can't check, an LLM-as-a-judge evaluator runs on demand.

To get started, deploy the showcase from the repository, change the system prompt, push to `main` to watch the release gate decide, and replace the rules with the ones your own agents must follow.

## Resources

- The action: [alvarog2491/agentcore-ab-release-gate](https://github.com/alvarog2491/agentcore-ab-release-gate)
- The showcase: [alvarog2491/agentcore-release-showcase](https://github.com/alvarog2491/agentcore-release-showcase)
- AWS documentation: [A/B testing](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ab-testing.html), [A/B tests with target-based routing](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ab-testing-target-based.html), [A/B testing prerequisites](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ab-testing-prereqs.html), [Code-based evaluators](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html), [Getting started with on-demand evaluation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-on-demand.html)
