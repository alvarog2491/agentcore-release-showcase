"""Deterministic AgentCore code-based evaluator: did the agent use its tools?

Runs as an AWS Lambda at TRACE level. AgentCore sends the session's ADOT spans
plus the trace under evaluation; the trace scores 1.0 (PASS) when it contains at
least one tool execution and 0.0 (FAIL) when the agent answered without tools.

Tool executions are the spans opentelemetry-instrumentation-langchain emits for
LangChain tools: gen_ai.operation.name = "execute_tool" (GenAI semantic
conventions) or traceloop.span.kind = "tool" (legacy attribute).

Contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html
Standard library only, so the Lambda needs no packaged dependencies.
"""


def _is_tool_span(span: dict) -> bool:
    attributes = span.get("attributes") or {}
    return (
        attributes.get("gen_ai.operation.name") == "execute_tool"
        or attributes.get("traceloop.span.kind") == "tool"
    )


def _is_agent_span(span: dict) -> bool:
    # Real spans carry a span name; log-event records only carry event attributes.
    return bool(span.get("name")) and "event.name" not in (span.get("attributes") or {})


def handler(event, context=None):
    spans = (event.get("evaluationInput") or {}).get("sessionSpans") or []
    trace_ids = (event.get("evaluationTarget") or {}).get("traceIds") or []

    # TRACE level scores one trace; without a target (SESSION level) score them all.
    if trace_ids:
        spans = [s for s in spans if s.get("traceId") in trace_ids]

    if not any(_is_agent_span(s) for s in spans):
        return {
            "errorCode": "NO_SPANS",
            "errorMessage": f"No agent spans found for trace(s) {trace_ids or 'in session'}.",
        }

    # A span can appear more than once; count each tool execution once.
    tool_calls = {}
    for span in spans:
        if _is_tool_span(span):
            name = (span.get("attributes") or {}).get("gen_ai.tool.name") or span.get("name", "unknown")
            tool_calls[span.get("spanId") or id(span)] = name

    if not tool_calls:
        return {
            "label": "FAIL",
            "value": 0.0,
            "explanation": "The agent answered without calling any tool.",
        }

    names = sorted(set(tool_calls.values()))
    return {
        "label": "PASS",
        "value": 1.0,
        "explanation": f"The agent made {len(tool_calls)} tool call(s): {', '.join(names)}.",
    }
