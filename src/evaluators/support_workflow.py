"""Deterministic AgentCore code-based evaluator: did the agent follow the support workflow?

Runs as an AWS Lambda at TRACE level. AgentCore sends the session's ADOT spans
plus the trace under evaluation (one agent turn). The turn scores 1.0 (PASS)
when it breaks none of the store's workflow rules and 0.0 (FAIL) otherwise:

1. Look it up. An answer that states order facts (order IDs, tracking numbers,
   RMA numbers, delivery or return dates) needs at least one tool call in the
   session so far. Asking the customer for missing information needs none.
2. No invented identifiers. Every order ID, tracking number and RMA number in
   the answer appears in the customer's messages or in a tool result.
3. Don't make the customer do the lookup. The agent never asks the customer
   for a SKU without having read the order with get_order_details: customers
   don't know SKUs, and the order lists them.
4. Check before acting. create_return for an item is called only after
   check_return_eligibility for the same order and SKU.

The spans come from opentelemetry-instrumentation-langchain: tool executions
(gen_ai.operation.name = "execute_tool") carry the tool name, arguments and
result; the LangGraph.workflow span carries the customer's message and the
agent's messages for the turn.

Contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html
Standard library only, so the Lambda needs no packaged dependencies.
"""

import json
import re

# The model often writes non-ASCII hyphens ("ORD\u20111001"); normalize them.
_HYPHENS = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")
_ORDER_ID = re.compile(r"\bORD-\d+\b")
_RMA = re.compile(r"\bRMA-\d+\b")
# Carrier tracking numbers: long uppercase alphanumeric tokens with digits (1Z..., JD...).
_TRACKING = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{12,}\b")
# A question in the answer that mentions a SKU: the agent asks the customer for one.
_ASKS_FOR_SKU = re.compile(r"[^.!?\n]*\bSKU\b[^.!?\n]*\?", re.IGNORECASE)
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}\b|\b\d{1,2}\s+(?:January|February|"
    r"March|April|May|June|July|August|September|October|November|December)\b"
)


def _normalize(text: str) -> str:
    return _HYPHENS.sub("-", text or "")


def _identifiers(text: str) -> set[str]:
    text = _normalize(text)
    return set(_ORDER_ID.findall(text)) | set(_RMA.findall(text)) | set(_TRACKING.findall(text))


def check_turn(customer_messages: list[str], tool_calls: list[dict], answer: str, turn_tool_calls: int) -> list[str]:
    """Return the workflow rules this turn breaks (empty when it passes).

    customer_messages: the customer's messages in the session up to this turn.
    tool_calls: the session's tool calls up to and including this turn, in
        order, as {"name", "inputs", "result"}.
    answer: the agent's final answer for this turn.
    turn_tool_calls: how many of tool_calls belong to this turn.
    """
    violations = []
    answer = _normalize(answer)
    mentioned = _identifiers(answer)
    from_customer = set()
    for message in customer_messages:
        from_customer |= _identifiers(message)

    # Repeating the customer's own order ID (for example, when asking for an
    # email) is not a fact the agent looked up.
    if (mentioned - from_customer or _DATE.search(answer)) and not tool_calls:
        violations.append("The answer states order facts but no tool was called to look them up.")

    known = set(from_customer)
    for call in tool_calls:
        known |= _identifiers(call.get("result", ""))
    invented = sorted(mentioned - known)
    if invented:
        violations.append(f"The answer contains identifiers no customer message or tool result supports: {', '.join(invented)}.")

    if _ASKS_FOR_SKU.search(answer) and not any(call.get("name") == "get_order_details" for call in tool_calls):
        violations.append("The agent asked the customer for a SKU instead of reading the order with get_order_details.")

    checked = set()
    for index, call in enumerate(tool_calls):
        inputs = call.get("inputs") or {}
        key = (_normalize(str(inputs.get("order_id", ""))).strip().upper(), str(inputs.get("sku", "")).strip().upper())
        if call.get("name") == "check_return_eligibility":
            checked.add(key)
        elif call.get("name") == "create_return" and key not in checked:
            if index >= len(tool_calls) - turn_tool_calls:
                violations.append(f"create_return was called for {key[0]} {key[1]} without check_return_eligibility first.")

    return violations


# --- Span parsing ---


def _load(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value or {}


def _message_text(message: dict) -> str:
    content = (message.get("kwargs") or {}).get("content", "")
    if isinstance(content, list):  # Converse content blocks: keep text, drop reasoning
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content or ""


def _message_type(message: dict) -> str:
    return (message.get("kwargs") or {}).get("type") or (message.get("id") or [""])[-1]


def _is_tool_span(span: dict) -> bool:
    attributes = span.get("attributes") or {}
    return attributes.get("gen_ai.operation.name") == "execute_tool" or attributes.get("traceloop.span.kind") == "tool"


def _tool_call(span: dict) -> dict:
    attributes = span.get("attributes") or {}
    arguments = _load(attributes.get("gen_ai.tool.call.arguments"))
    result = _load(attributes.get("gen_ai.tool.call.result"))
    output = result.get("output") if isinstance(result, dict) else None
    return {
        "name": attributes.get("gen_ai.tool.name") or span.get("name", "").removeprefix("execute_tool "),
        "inputs": arguments.get("inputs") or {} if isinstance(arguments, dict) else {},
        "result": _message_text(output) if isinstance(output, dict) else str(output or ""),
    }


def _turns(spans: list[dict]) -> list[dict]:
    """One record per agent turn (trace), in time order."""
    by_trace = {}
    for span in spans:
        if not span.get("name") or "event.name" in (span.get("attributes") or {}):
            continue  # log-event records, not spans
        by_trace.setdefault(span.get("traceId"), []).append(span)

    turns = []
    for trace_id, trace_spans in by_trace.items():
        trace_spans.sort(key=lambda s: s.get("startTimeUnixNano", 0))
        tool_calls, seen = [], set()
        for span in trace_spans:
            if _is_tool_span(span) and span.get("spanId") not in seen:  # a span can appear twice
                seen.add(span.get("spanId"))
                tool_calls.append(_tool_call(span))
        workflow = next((s for s in trace_spans if s.get("name") == "LangGraph.workflow"), None)
        customer, answer = "", None
        if workflow:
            attributes = workflow.get("attributes") or {}
            inputs = (_load(attributes.get("gen_ai.task.input")).get("inputs") or {}).get("messages") or []
            customer = " ".join(_message_text(m) for m in inputs if _message_type(m) in ("human", "HumanMessage"))
            outputs = (_load(attributes.get("gen_ai.task.output")).get("outputs") or {}).get("messages") or []
            replies = [m for m in outputs if _message_type(m) in ("ai", "AIMessage")]
            if replies:
                answer = _message_text(replies[-1])
        start = min(s.get("startTimeUnixNano", 0) for s in trace_spans)
        turns.append({"trace_id": trace_id, "start": start, "customer": customer, "answer": answer, "tool_calls": tool_calls})
    return sorted(turns, key=lambda t: t["start"])


def handler(event, context=None):
    spans = (event.get("evaluationInput") or {}).get("sessionSpans") or []
    trace_ids = (event.get("evaluationTarget") or {}).get("traceIds") or []
    turns = _turns(spans)

    # TRACE level scores one turn; without a target (SESSION level) score the last one.
    candidates = [i for i, t in enumerate(turns) if t["answer"] is not None and (not trace_ids or t["trace_id"] in trace_ids)]
    target = candidates[-1] if candidates else None
    if target is None:
        return {
            "errorCode": "NO_SPANS",
            "errorMessage": f"No agent turn found for trace(s) {trace_ids or 'in session'}.",
        }

    history = turns[: target + 1]
    turn = turns[target]
    violations = check_turn(
        customer_messages=[t["customer"] for t in history],
        tool_calls=[call for t in history for call in t["tool_calls"]],
        answer=turn["answer"],
        turn_tool_calls=len(turn["tool_calls"]),
    )
    if violations:
        return {"label": "FAIL", "value": 0.0, "explanation": " ".join(violations)}

    names = ", ".join(call["name"] for call in turn["tool_calls"]) or "none"
    return {"label": "PASS", "value": 1.0, "explanation": f"The turn follows the support workflow (tool calls: {names})."}
