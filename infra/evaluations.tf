# Deterministic tool-usage evaluation: a code-based evaluator backed by a Lambda
# (src/evaluators/tool_usage.py) that scores each agent turn 1.0 if it called at
# least one tool and 0.0 if it did not. No LLM judge involved.
# Plus the online evaluation config the release gate uses as its template, which
# also scores every session with the LLM-as-a-judge evaluators (evaluators.tf)
# and three built-in evaluators.

locals {
  tool_usage_evaluator_name = "${var.agent_name}_tool_usage"
}

# --- Evaluator Lambda ---

data "archive_file" "tool_usage_evaluator" {
  type        = "zip"
  source_file = "${path.module}/../src/evaluators/tool_usage.py"
  output_path = "${path.module}/.build/tool_usage_evaluator.zip"
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "tool_usage_evaluator" {
  name               = "${var.project_name}-tool-usage-evaluator"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "tool_usage_evaluator_logs" {
  role       = aws_iam_role.tool_usage_evaluator.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "tool_usage_evaluator" {
  name              = "/aws/lambda/${var.project_name}-tool-usage-evaluator"
  retention_in_days = 30
}

resource "aws_lambda_function" "tool_usage_evaluator" {
  function_name    = "${var.project_name}-tool-usage-evaluator"
  description      = "AgentCore code-based evaluator: did the agent call a tool in this turn?"
  role             = aws_iam_role.tool_usage_evaluator.arn
  runtime          = "python3.13"
  architectures    = ["arm64"]
  handler          = "tool_usage.handler"
  filename         = data.archive_file.tool_usage_evaluator.output_path
  source_code_hash = data.archive_file.tool_usage_evaluator.output_base64sha256
  timeout          = 30
  memory_size      = 256

  depends_on = [
    aws_iam_role_policy_attachment.tool_usage_evaluator_logs,
    aws_cloudwatch_log_group.tool_usage_evaluator,
  ]
}

# Only AgentCore evaluators in this account may invoke the function.
resource "aws_lambda_permission" "agentcore_evaluations" {
  statement_id   = "AllowAgentCoreEvaluations"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.tool_usage_evaluator.function_name
  principal      = "bedrock-agentcore.amazonaws.com"
  source_account = local.account_id
  source_arn     = "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:evaluator/*"
}

# --- Evaluator ---

resource "aws_bedrockagentcore_evaluator" "tool_usage" {
  evaluator_name = local.tool_usage_evaluator_name
  description    = "Deterministic: 1.0 if the agent called at least one tool in the turn, else 0.0."
  level          = "TRACE"

  evaluator_config {
    code_based {
      lambda_config {
        lambda_arn                = aws_lambda_function.tool_usage_evaluator.arn
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

  # An online evaluation config holds at most 10 evaluators: tool usage, the six
  # custom judges and three built-ins. Every evaluator in the workflow's
  # quality-gates must be here; the others are recorded for monitoring.
  evaluator {
    evaluator_id = aws_bedrockagentcore_evaluator.tool_usage.evaluator_id
  }

  dynamic "evaluator" {
    for_each = aws_bedrockagentcore_evaluator.judge
    content {
      evaluator_id = evaluator.value.evaluator_id
    }
  }

  dynamic "evaluator" {
    for_each = var.monitored_builtin_evaluators
    content {
      evaluator_id = evaluator.value
    }
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
