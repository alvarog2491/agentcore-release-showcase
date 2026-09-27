variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Project name used for naming and tagging"
  type        = string
  default     = "agentcore-release-showcase"
}

variable "agent_name" {
  description = "AgentCore Runtime name. Letters, digits and underscores only."
  type        = string
  default     = "showcase_agent"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]{0,47}$", var.agent_name))
    error_message = "agent_name must start with a letter and contain only letters, digits and underscores (max 48 chars)."
  }
}

variable "initial_image_tag" {
  description = "ECR image tag used to create the Runtime. Later versions are rolled out by the release gate action, not Terraform."
  type        = string
  default     = "bootstrap"
}

variable "control_endpoint_name" {
  description = "Name of the stable Runtime endpoint and its Gateway target. Must match the action's control-endpoint-name input."
  type        = string
  default     = "control"
}

variable "bedrock_model_id" {
  description = "Bedrock model the agent invokes. Must match MODEL_ID in src/agent/model/load.py."
  type        = string
  default     = "openai.gpt-oss-20b-1:0"
}

variable "judge_model_id" {
  description = "Bedrock foundation model ID the custom LLM-as-a-judge evaluators use. A different, larger model than the agent's."
  type        = string
  default     = "openai.gpt-oss-120b-1:0"
}


variable "monitored_builtin_evaluators" {
  description = "Built-in evaluators added to the online evaluation config for monitoring. The config holds at most 10 evaluators, 7 of them custom."
  type        = list(string)
  default     = ["Builtin.Helpfulness", "Builtin.Faithfulness", "Builtin.ToolSelectionAccuracy"]
}

variable "evaluation_sampling_percentage" {
  description = "Percentage of sessions the online evaluation scores. 100 keeps the showcase's small traffic fully scored."
  type        = number
  default     = 100
}

variable "evaluation_session_timeout_minutes" {
  description = "Idle minutes after which a session counts as complete and gets scored."
  type        = number
  default     = 2
}

variable "github_oidc_sub_prefix" {
  description = <<-EOT
    Prefix of the GitHub OIDC token "sub" claim for the repository whose release
    workflow may assume the deploy role. With GitHub's immutable subjects it
    includes owner and repository IDs; read it with:
    gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
  EOT
  type        = string
  default     = "repo:alvarog2491@159990212/agentcore-release-showcase@1391038404"
}
