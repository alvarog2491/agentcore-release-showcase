# LLM-as-a-judge evaluators, following a three-layer approach: built-in
# evaluators for general quality (no resources needed), custom evaluators for
# business accuracy, and explainability evaluators for trust and auditability.
# All are TRACE level: one score per agent turn.
#
# All six are in the release gate's online evaluation config (evaluations.tf),
# so every A/B release scores both variants with them. The workflow's
# quality-gates gate on the business-accuracy ones (return_policy,
# order_grounding); the explainability scores are recorded but don't block a
# release. They also run on demand with scripts/evaluate.py.
#
# Trace-level instructions must use the {context} placeholder (previous turns,
# the current user prompt and the current tool calls) and {assistant_turn}
# (the agent's answer for the turn being scored).

locals {
  # Facts the judges need to check the agent against: the store's return
  # policy and the fixed "today" of the store dataset (src/agent/tools/store.py).
  store_policy = <<-EOT
    Store return policy (today is 2026-09-27):
    - An item can be returned within 30 days of its delivery date.
    - Items marked "final sale" (for example gift cards) cannot be returned.
    - Orders that have not been delivered yet cannot be returned.
    - A return only exists once the create_return tool has returned an RMA number.
  EOT

  # Same three-point scale for every custom evaluator, so scores are comparable.
  judge_scale = {
    no      = 0.0
    partial = 0.5
    yes     = 1.0
  }

  judge_evaluators = {
    # --- Layer 2: business accuracy ---

    return_policy = {
      description = "Business accuracy: return decisions follow the store's return policy and the tool results."
      question    = <<-EOT
        Did the agent apply the store return policy correctly in this turn?
        Check that:
        - it only told the customer an item is returnable (or not) after a
          check_return_eligibility or create_return result supports it,
        - it never created, or claimed to have created, a return for an
          item the policy excludes,
        - every RMA number in the answer comes from a create_return result.
        If the turn has nothing to do with returns, answer Yes.
      EOT
      yes         = "Every return statement follows the policy and is backed by a tool result, or the turn is not about returns."
      partial     = "The return decision is correct, but part of it (a date, a refund amount, a next step) is not backed by a tool result."
      no          = "The agent promised, created or invented a return that the policy or the tool results do not allow."
    }

    order_grounding = {
      description = "Business accuracy: order facts in the answer come from tool results, not from the model."
      question    = <<-EOT
        Are the order facts in the agent's answer grounded in the tool results
        shown in the context? Order facts are order IDs, items, SKUs, prices,
        statuses, carriers, tracking numbers and dates.
        A fact that appears in no tool result is invented, even if it sounds
        plausible. Asking the customer for missing information is grounded.
      EOT
      yes         = "Every order fact in the answer matches a tool result, or the answer contains no order facts."
      partial     = "Most order facts match the tool results, but at least one is missing from them or slightly changed."
      no          = "The answer states order facts that do not appear in any tool result or contradict them."
    }

    # --- Layer 3: explainability ---

    decision_rationale = {
      description = "Explainability: the agent explains why it reached its answer or decision."
      question    = <<-EOT
        Does the agent explain why it reached its answer? A good rationale
        connects the outcome to the reason for it, for example "the headphones
        can be returned because they were delivered on 2026-09-14 and the
        30-day window is still open", not only "yes, you can return them".
      EOT
      yes         = "The answer states the outcome and the reason that led to it."
      partial     = "The answer hints at a reason, but the link between the reason and the outcome is unclear or incomplete."
      no          = "The answer states an outcome with no reason at all."
    }

    evidence_attribution = {
      description = "Explainability: the agent cites the data it used (order record, tracking events, eligibility result)."
      question    = <<-EOT
        Does the agent point the customer to the evidence behind its answer?
        Good evidence attribution names the record or data the answer is based
        on, for example "according to the DHL tracking for ORD-1002, the
        parcel arrived at the Lyon sorting facility on 2026-09-26", so the
        customer could verify it.
      EOT
      yes         = "The answer names the specific data it is based on (order, shipment events, eligibility result)."
      partial     = "The answer mentions where the information comes from only in general terms."
      no          = "The answer gives information without saying where it comes from."
    }

    policy_reasoning = {
      description = "Explainability: the agent explains which policy constraints shaped the answer."
      question    = <<-EOT
        When a store policy constraint shaped the answer (the 30-day return
        window, final-sale items, returns only after delivery), does the agent
        say which constraint applied and how? For example "gift cards are final
        sale, so they cannot be returned" or "your return window is open until
        2026-10-14". If no policy constraint was relevant to the turn, answer Yes.
      EOT
      yes         = "The answer names the constraint that applied and how it affected the outcome, or no constraint was relevant."
      partial     = "The answer mentions a constraint but does not explain how it affected the outcome."
      no          = "A policy constraint decided the outcome, but the answer does not mention it."
    }

    assumption_disclosure = {
      description = "Explainability: the agent states its assumptions or asks when information is missing."
      question    = <<-EOT
        When the customer's request was incomplete or ambiguous (no email,
        no order ID, an item that matches more than one order, an unknown
        customer), did the agent say what it assumed or ask for the missing
        information, instead of silently guessing? If the request was complete
        and unambiguous, answer Yes.
      EOT
      yes         = "The agent stated its assumptions or asked for the missing information, or nothing was missing."
      partial     = "The agent made a reasonable assumption but did not say so clearly."
      no          = "The agent silently guessed missing information and presented the guess as fact."
    }
  }
}

resource "aws_bedrockagentcore_evaluator" "judge" {
  for_each = local.judge_evaluators

  evaluator_name = "${var.agent_name}_${each.key}"
  description    = each.value.description
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

        ${each.value.question}
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
          value      = local.judge_scale.yes
          definition = each.value.yes
        }
        numerical {
          label      = "Partially"
          value      = local.judge_scale.partial
          definition = each.value.partial
        }
        numerical {
          label      = "No"
          value      = local.judge_scale.no
          definition = each.value.no
        }
      }
    }
  }
}
