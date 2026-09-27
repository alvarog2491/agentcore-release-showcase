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
