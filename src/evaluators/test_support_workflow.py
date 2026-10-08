"""Tests for the support-workflow evaluator. Run: python -m unittest discover src/evaluators"""

import json
import unittest

from support_workflow import check_turn, handler

ELIGIBLE = "Wireless Headphones can be returned until 2026-10-14 for a refund of 129.00 EUR."
CREATED = "Return RMA-5001 created for Wireless Headphones (reason: hurts)."
ORDER = "Order ORD-1001 placed 2026-09-10, status delivered.\n- HD-200 Wireless Headphones: 129.00 EUR"


def call(name, result, **inputs):
    return {"name": name, "inputs": inputs, "result": result}


class CheckTurnTest(unittest.TestCase):
    def test_checked_return_passes(self):
        calls = [
            call("check_return_eligibility", ELIGIBLE, order_id="ORD-1001", sku="HD-200"),
            call("create_return", CREATED, order_id="ORD-1001", sku="HD-200", reason="hurts"),
        ]
        answer = "Your return RMA‑5001 is open; you can send them back until 2026-10-14."
        self.assertEqual(check_turn(["Return the headphones from ORD-1001"], calls, answer, 2), [])

    def test_return_without_eligibility_check_fails(self):
        calls = [call("create_return", CREATED, order_id="ORD-1001", sku="HD-200", reason="hurts")]
        violations = check_turn(["Return the headphones from ORD-1001"], calls, "Return RMA-5001 is open.", 1)
        self.assertEqual(len(violations), 1)
        self.assertIn("without check_return_eligibility", violations[0])

    def test_check_in_an_earlier_turn_counts(self):
        calls = [
            call("check_return_eligibility", ELIGIBLE, order_id="ORD-1001", sku="HD-200"),
            call("create_return", CREATED, order_id="ORD-1001", sku="HD-200", reason="hurts"),
        ]
        self.assertEqual(check_turn(["Can I return them?", "Yes, please"], calls, "Done: RMA-5001.", 1), [])

    def test_invented_identifier_fails(self):
        calls = [call("get_order_details", ORDER, order_id="ORD-1001")]
        violations = check_turn(["What's in ORD-1001?"], calls, "Your tracking number is 1Z999AA10123456784.", 1)
        self.assertIn("1Z999AA10123456784", violations[0])

    def test_facts_without_lookup_fail(self):
        violations = check_turn(["When will ORD-1002 arrive?"], [], "It arrives on 2026-09-29.", 0)
        self.assertIn("no tool was called", violations[0])

    def test_clarifying_question_passes(self):
        answer = "Could you share the email address you used for order ORD‑1004?"
        self.assertEqual(check_turn(["Please return the gift card in ORD-1004"], [], answer, 0), [])

    def test_asking_for_sku_fails(self):
        answer = "Sure! Could you tell me the SKU of the headphones in ORD-1001?"
        violations = check_turn(["Return the headphones from ORD-1001"], [], answer, 0)
        self.assertIn("asked the customer for a SKU", violations[0])

    def test_asking_to_confirm_a_sku_after_reading_the_order_passes(self):
        calls = [call("get_order_details", ORDER, order_id="ORD-1001")]
        answer = "Do you mean the HD-200 headphones (SKU HD-200)?"
        self.assertEqual(check_turn(["Return my headphones from ORD-1001"], calls, answer, 1), [])


def span(trace, span_id, name, start, **attributes):
    return {"traceId": trace, "spanId": span_id, "name": name, "startTimeUnixNano": start, "attributes": attributes}


def message(kind, content):
    return {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", kind], "kwargs": {"content": content}}


def turn_spans(trace, start, prompt, tools, answer):
    spans = [
        span(
            trace, f"{trace}-wf", "LangGraph.workflow", start,
            **{
                "gen_ai.task.input": json.dumps({"inputs": {"messages": [message("HumanMessage", prompt)]}}),
                "gen_ai.task.output": json.dumps({"outputs": {"messages": [message("AIMessage", answer)]}}),
            },
        )
    ]
    for i, (name, inputs, result) in enumerate(tools, start=1):
        spans.append(
            span(
                trace, f"{trace}-t{i}", f"execute_tool {name}", start + i,
                **{
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": name,
                    "gen_ai.tool.call.arguments": json.dumps({"inputs": inputs}),
                    "gen_ai.tool.call.result": json.dumps({"output": message("ToolMessage", result)}),
                },
            )
        )
    return spans


class HandlerTest(unittest.TestCase):
    def test_scores_the_target_turn_with_session_history(self):
        spans = turn_spans(
            "t1", 1, "Can I return the headphones from ORD-1001?",
            [("check_return_eligibility", {"order_id": "ORD-1001", "sku": "HD-200"}, ELIGIBLE)],
            "Yes, until 2026-10-14.",
        ) + turn_spans(
            "t2", 100, "Yes please, open it.",
            [("create_return", {"order_id": "ORD-1001", "sku": "HD-200", "reason": "hurts"}, CREATED)],
            "Done: RMA-5001.",
        )
        event = {"evaluationInput": {"sessionSpans": spans}, "evaluationTarget": {"traceIds": ["t2"]}}
        self.assertEqual(handler(event)["label"], "PASS")

    def test_fails_a_blind_return(self):
        spans = turn_spans(
            "t1", 1, "Return the headphones from ORD-1001.",
            [
                ("get_order_details", {"order_id": "ORD-1001"}, ORDER),
                ("create_return", {"order_id": "ORD-1001", "sku": "HD-200", "reason": "hurts"}, CREATED),
            ],
            "Done: RMA-5001.",
        )
        result = handler({"evaluationInput": {"sessionSpans": spans}, "evaluationTarget": {"traceIds": ["t1"]}})
        self.assertEqual((result["label"], result["value"]), ("FAIL", 0.0))

    def test_no_agent_turn_is_an_error(self):
        result = handler({"evaluationInput": {"sessionSpans": []}, "evaluationTarget": {"traceIds": ["x"]}})
        self.assertEqual(result["errorCode"], "NO_SPANS")


if __name__ == "__main__":
    unittest.main()
