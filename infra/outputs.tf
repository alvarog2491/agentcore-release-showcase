output "ecr_repository_url" {
  description = "Push agent images here"
  value       = aws_ecr_repository.agent.repository_url
}

output "runtime_id" {
  description = "Action input: runtime-id"
  value       = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_id
}

output "runtime_arn" {
  value = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn
}

output "control_endpoint_name" {
  description = "Action input: control-endpoint-name"
  value       = aws_bedrockagentcore_agent_runtime_endpoint.control.name
}

output "gateway_id" {
  description = "Action input: gateway-id"
  value       = aws_bedrockagentcore_gateway.agent.gateway_id
}

output "gateway_url" {
  description = "Send agent traffic here during the observation period"
  value       = aws_bedrockagentcore_gateway.agent.gateway_url
}

output "aws_region" {
  description = "Action input: aws-region"
  value       = var.aws_region
}

output "evaluation_config_id" {
  description = "Action input: evaluation-config-id"
  value       = aws_bedrockagentcore_online_evaluation_config.control.online_evaluation_config_id
}

output "support_workflow_evaluator_id" {
  description = "Release gate evaluator ID (key in the workflow's quality gates)"
  value       = aws_bedrockagentcore_evaluator.support_workflow.evaluator_id
}

output "order_grounding_evaluator_id" {
  description = "On-demand LLM-as-a-judge evaluator ID for scripts/evaluate.py"
  value       = aws_bedrockagentcore_evaluator.order_grounding.evaluator_id
}

output "ab_test_role_arn" {
  description = "Action input: ab-test-role-arn"
  value       = aws_iam_role.ab_test.arn
}

output "github_deploy_role_arn" {
  description = "Repository variable AWS_DEPLOY_ROLE_ARN"
  value       = aws_iam_role.github_deploy.arn
}
