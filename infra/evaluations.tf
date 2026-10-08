# Deterministic release gate: a code-based evaluator backed by a Lambda
# (src/evaluators/support_workflow.py) that scores each agent turn 1.0 if it
# follows the store's support workflow and 0.0 if it breaks a rule. No LLM
# judge involved. Plus the online evaluation config the release gate uses as
# its template.

locals {
  support_workflow_evaluator_name = "${var.agent_name}_support_workflow"
}

# --- Evaluator Lambda ---

data "archive_file" "support_workflow_evaluator" {
  type        = "zip"
  source_file = "${path.module}/../src/evaluators/support_workflow.py"
  output_path = "${path.module}/.build/support_workflow_evaluator.zip"
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

resource "aws_iam_role" "support_workflow_evaluator" {
  name               = "${var.project_name}-support-workflow-evaluator"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "support_workflow_evaluator_logs" {
  role       = aws_iam_role.support_workflow_evaluator.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "support_workflow_evaluator" {
  name              = "/aws/lambda/${var.project_name}-support-workflow-evaluator"
  retention_in_days = 30
}

resource "aws_lambda_function" "support_workflow_evaluator" {
  function_name    = "${var.project_name}-support-workflow-evaluator"
  description      = "AgentCore code-based evaluator: did the agent follow the support workflow in this turn?"
  role             = aws_iam_role.support_workflow_evaluator.arn
  runtime          = "python3.13"
  architectures    = ["arm64"]
  handler          = "support_workflow.handler"
  filename         = data.archive_file.support_workflow_evaluator.output_path
  source_code_hash = data.archive_file.support_workflow_evaluator.output_base64sha256
  timeout          = 30
  memory_size      = 256

  depends_on = [
    aws_iam_role_policy_attachment.support_workflow_evaluator_logs,
    aws_cloudwatch_log_group.support_workflow_evaluator,
  ]
}

# Only AgentCore evaluators in this account may invoke the function.
resource "aws_lambda_permission" "agentcore_evaluations" {
  statement_id   = "AllowAgentCoreEvaluations"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.support_workflow_evaluator.function_name
  principal      = "bedrock-agentcore.amazonaws.com"
  source_account = local.account_id
  source_arn     = "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:evaluator/*"
}

# --- Evaluator ---

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
