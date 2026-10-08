data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
}

data "aws_iam_policy_document" "agentcore_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:*"]
    }
  }
}

# --- Runtime execution role ---

resource "aws_iam_role" "runtime" {
  name               = "${var.project_name}-runtime"
  assume_role_policy = data.aws_iam_policy_document.agentcore_assume_role.json
}

data "aws_iam_policy_document" "runtime" {
  statement {
    sid       = "EcrToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = [aws_ecr_repository.agent.arn]
  }

  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"]
  }

  statement {
    sid       = "LogsDescribe"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:*"]
  }

  # Required by the release gate for online evaluation telemetry.
  statement {
    sid = "XRay"
    actions = [
      "xray:PutTraceSegments",
      "xray:PutTelemetryRecords",
      "xray:GetSamplingRules",
      "xray:GetSamplingTargets",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["bedrock-agentcore"]
    }
  }

  statement {
    sid = "WorkloadIdentity"
    actions = [
      "bedrock-agentcore:GetWorkloadAccessToken",
      "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
      "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
    ]
    resources = [
      "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:workload-identity-directory/default",
      "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:workload-identity-directory/default/workload-identity/${var.agent_name}-*",
    ]
  }

  statement {
    sid = "BedrockModel"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = ["arn:${local.partition}:bedrock:${var.aws_region}::foundation-model/${var.bedrock_model_id}"]
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "runtime"
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

# --- Gateway role: lets the gateway invoke the runtime's endpoints ---

resource "aws_iam_role" "gateway" {
  name               = "${var.project_name}-gateway"
  assume_role_policy = data.aws_iam_policy_document.agentcore_assume_role.json
}

data "aws_iam_policy_document" "gateway" {
  statement {
    sid     = "InvokeRuntime"
    actions = ["bedrock-agentcore:InvokeAgentRuntime"]
    # The runtime ARN plus its endpoints (control, and treatment created by the action).
    resources = [
      aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn,
      "${aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "gateway" {
  name   = "invoke-runtime"
  role   = aws_iam_role.gateway.id
  policy = data.aws_iam_policy_document.gateway.json
}

# --- Evaluation execution role: reads agent traces, runs the evaluator Lambda ---

data "aws_iam_policy_document" "evaluation_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values = [
        "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:evaluator/*",
        "arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:online-evaluation-config/*",
      ]
    }
  }
}

resource "aws_iam_role" "evaluation" {
  name               = "${var.project_name}-evaluation"
  assume_role_policy = data.aws_iam_policy_document.evaluation_assume_role.json
}

data "aws_iam_policy_document" "evaluation" {
  statement {
    sid       = "ReadTraces"
    actions   = ["logs:DescribeLogGroups", "logs:StartQuery", "logs:GetQueryResults"]
    resources = ["*"]
  }

  statement {
    sid       = "WriteResults"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/bedrock-agentcore/evaluations/*"]
  }

  statement {
    sid     = "SpanIndexing"
    actions = ["logs:DescribeIndexPolicies", "logs:PutIndexPolicy"]
    resources = [
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:aws/spans",
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:aws/spans:*",
    ]
  }

  # Code-based evaluator Lambda.
  statement {
    sid       = "InvokeEvaluatorLambda"
    actions   = ["lambda:InvokeFunction", "lambda:GetFunction"]
    resources = [aws_lambda_function.support_workflow_evaluator.arn]
  }
}

resource "aws_iam_role_policy" "evaluation" {
  name   = "evaluation"
  role   = aws_iam_role.evaluation.id
  policy = data.aws_iam_policy_document.evaluation.json
}
