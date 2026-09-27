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
