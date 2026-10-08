# On-demand LLM-as-a-judge evaluator for the facts the release gate's rules
# can't check: is every order fact in the answer (dates, prices, statuses,
# weekdays) backed by a tool result? TRACE level: one score per agent turn.
#
# Not in the online evaluation config: it runs only with scripts/evaluate.py.
# Adding it there would lock it and add judge cost to every scored session.
#
# Trace-level instructions must use the {context} placeholder (previous turns,
# the current user prompt and the current tool calls) and {assistant_turn}
# (the agent's answer for the turn being scored).

# Kept the evaluator ID when the other judges were removed.
moved {
  from = aws_bedrockagentcore_evaluator.judge["order_grounding"]
  to   = aws_bedrockagentcore_evaluator.order_grounding
}

locals {
  # The store's return policy and the fixed "today" of the store dataset
  # (src/agent/tools/store.py). Keep in sync with the tools.
  store_policy = <<-EOT
    Store return policy (today is 2026-09-27):
    - An item can be returned within 30 days of its delivery date.
    - Items marked "final sale" (for example gift cards) cannot be returned.
    - Orders that have not been delivered yet cannot be returned.
    - A return only exists once the create_return tool has returned an RMA number.
  EOT

  order_grounding_question = <<-EOT
    Are the order facts in the agent's answer grounded in the tool results
    shown in the context? Order facts are order IDs, items, SKUs, prices,
    statuses, carriers, tracking numbers and dates.
    A fact that appears in no tool result is invented, even if it sounds
    plausible. Asking the customer for missing information is grounded.
  EOT
}

resource "aws_bedrockagentcore_evaluator" "order_grounding" {
  evaluator_name = "${var.agent_name}_order_grounding"
  description    = "Business accuracy: order facts in the answer come from tool results, not from the model."
  level          = "TRACE"

  evaluator_config {
    llm_as_a_judge {
      instructions = <<-EOT
        You are evaluating one turn of a customer-support agent for an online
        electronics store. The agent can call these tools: find_customer_orders,
        get_order_details, track_shipment, check_return_eligibility and
        create_return. Tool results are the only source of truth about orders.

        ${local.store_policy}
        ## Conversation so far, including the tool calls and results of this turn

        {context}

        ## Agent answer to evaluate

        {assistant_turn}

        ## Evaluation question

        ${local.order_grounding_question}
      EOT

      model_config {
        bedrock_evaluator_model_config {
          model_id = var.judge_model_id

          inference_config {
            max_tokens  = 1024
            temperature = 0
          }
        }
      }

      rating_scale {
        numerical {
          label      = "Yes"
          value      = 1.0
          definition = "Every order fact in the answer matches a tool result, or the answer contains no order facts."
        }
        numerical {
          label      = "Partially"
          value      = 0.5
          definition = "Most order facts match the tool results, but at least one is missing from them or slightly changed."
        }
        numerical {
          label      = "No"
          value      = 0.0
          definition = "The answer states order facts that do not appear in any tool result or contradict them."
        }
      }
    }
  }
}
