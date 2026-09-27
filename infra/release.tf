# Roles used by the release: the A/B test execution role (assumed by AgentCore)
# and the GitHub Actions deploy role (assumed through OIDC by release.yml).

# --- A/B test execution role ---

data "aws_iam_policy_document" "ab_test_assume_role" {
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
      values   = ["arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:ab-test/*"]
    }
  }
}

resource "aws_iam_role" "ab_test" {
  name               = "${var.project_name}-ab-test"
  assume_role_policy = data.aws_iam_policy_document.ab_test_assume_role.json
}

data "aws_iam_policy_document" "ab_test" {
  statement {
    sid = "AgentCoreResources"
    actions = [
      "bedrock-agentcore:GetGateway",
      "bedrock-agentcore:GetGatewayTarget",
      "bedrock-agentcore:ListGatewayTargets",
      "bedrock-agentcore:CreateGatewayRule",
      "bedrock-agentcore:UpdateGatewayRule",
      "bedrock-agentcore:GetGatewayRule",
      "bedrock-agentcore:DeleteGatewayRule",
      "bedrock-agentcore:ListGatewayRules",
      "bedrock-agentcore:GetOnlineEvaluationConfig",
      "bedrock-agentcore:GetEvaluator",
    ]
    resources = ["arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:*"]
  }

  statement {
    sid       = "CloudWatchLogsDescribe"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["*"]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:DescribeIndexPolicies",
      "logs:PutIndexPolicy",
      "logs:StartQuery",
      "logs:GetQueryResults",
      "logs:StopQuery",
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
    ]
    resources = [
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/bedrock-agentcore/evaluations/*",
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:aws/spans",
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:aws/spans:*",
    ]
  }
}

resource "aws_iam_role_policy" "ab_test" {
  name   = "ab-test"
  role   = aws_iam_role.ab_test.id
  policy = data.aws_iam_policy_document.ab_test.json
}

# --- GitHub Actions deploy role (OIDC) ---

# The account already has the GitHub OIDC provider; there can only be one per URL.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # publish/traffic jobs run on main; the deploy job runs in the production environment.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "${var.github_oidc_sub_prefix}:ref:refs/heads/main",
        "${var.github_oidc_sub_prefix}:environment:production",
      ]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.project_name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_assume_role.json
  # The release gate keeps credentials for observation + evaluation (up to 4h).
  max_session_duration = 14400
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPushAndDescribe"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.agent.arn]
  }

  # Permissions the release gate action documents, scoped to this project's resources.
  statement {
    sid = "ReleaseGateRuntime"
    actions = [
      "bedrock-agentcore:GetAgentRuntime",
      "bedrock-agentcore:UpdateAgentRuntime",
      "bedrock-agentcore:GetAgentRuntimeEndpoint",
      "bedrock-agentcore:CreateAgentRuntimeEndpoint",
      "bedrock-agentcore:UpdateAgentRuntimeEndpoint",
    ]
    resources = [
      aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn,
      "${aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn}/*",
    ]
  }

  statement {
    sid = "ReleaseGateGateway"
    actions = [
      "bedrock-agentcore:GetGateway",
      "bedrock-agentcore:ListGatewayTargets",
      "bedrock-agentcore:GetGatewayTarget",
      "bedrock-agentcore:CreateGatewayTarget",
      "bedrock-agentcore:InvokeGateway", # traffic generation during observation
    ]
    resources = [
      aws_bedrockagentcore_gateway.agent.gateway_arn,
      "${aws_bedrockagentcore_gateway.agent.gateway_arn}/*",
    ]
  }

  statement {
    sid = "ReleaseGateAbTestAndEvaluation"
    actions = [
      "bedrock-agentcore:CreateABTest",
      "bedrock-agentcore:GetABTest",
      "bedrock-agentcore:UpdateABTest",
      "bedrock-agentcore:ListABTests",
      "bedrock-agentcore:GetOnlineEvaluationConfig",
      "bedrock-agentcore:CreateOnlineEvaluationConfig",
      "bedrock-agentcore:DeleteOnlineEvaluationConfig",
      "bedrock-agentcore:GetEvaluator",
    ]
    resources = ["arn:${local.partition}:bedrock-agentcore:${var.aws_region}:${local.account_id}:*"]
  }

  # Creating online evaluation configs sets up span indexing in CloudWatch Logs.
  statement {
    sid       = "EvaluationLogs"
    actions   = ["logs:DescribeIndexPolicies", "logs:PutIndexPolicy", "logs:CreateLogGroup"]
    resources = ["*"]
  }

  statement {
    sid     = "PassRoles"
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.runtime.arn,
      aws_iam_role.ab_test.arn,
      aws_iam_role.evaluation.arn,
    ]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["bedrock-agentcore.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "release"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}
