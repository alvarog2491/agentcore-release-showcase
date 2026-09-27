"""Run AgentCore on-demand evaluations against one agent session.

Downloads the session's spans from CloudWatch (the aws/spans log group and the
runtime endpoint's log group), then calls the Evaluate API once per evaluator,
targeting every turn (TRACE level), every tool call (TOOL_CALL level) or the
whole session (SESSION level) as the evaluator requires. Results are printed
and saved as a Markdown report under results/<session-id>/<timestamp>/.

Evaluators can be built-in IDs (Builtin.Helpfulness), custom evaluator IDs
(showcase_agent_return_policy-AbCdEf1234) or custom evaluator names
(showcase_agent_return_policy), which are resolved to their IDs.

Usage:
    uv run scripts/evaluate.py --runtime-id <runtime-id> --session-id <session-id> \
        --evaluators Builtin.Helpfulness,showcase_agent_return_policy
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

# Evaluate accepts at most 10 trace or span IDs per call.
MAX_TARGETS = 10


def query_session_logs(logs, log_group, session_id, lookback_minutes):
    now = datetime.now(timezone.utc)
    query_id = logs.start_query(
        logGroupName=log_group,
        startTime=int((now - timedelta(minutes=lookback_minutes)).timestamp()),
        endTime=int(now.timestamp()),
        queryString=f"""fields @timestamp, @message
            | filter ispresent(scope.name) and attributes.session.id = "{session_id}"
            | sort @timestamp asc
            | limit 10000""",
    )["queryId"]
    while (result := logs.get_query_results(queryId=query_id))["status"] in ("Scheduled", "Running"):
        time.sleep(1)
    if result["status"] != "Complete":
        raise RuntimeError(f"CloudWatch Logs query on {log_group} ended with status {result['status']}")
    return [
        json.loads(field["value"])
        for row in result["results"]
        for field in row
        if field["field"] == "@message" and field["value"].lstrip().startswith("{")
    ]


def download_session_spans(region, runtime_id, endpoint, session_id, lookback_minutes):
    logs = boto3.client("logs", region_name=region)
    spans = query_session_logs(logs, "aws/spans", session_id, lookback_minutes)
    events = query_session_logs(
        logs, f"/aws/bedrock-agentcore/runtimes/{runtime_id}-{endpoint}", session_id, lookback_minutes
    )
    print(f"Downloaded {len(spans)} spans and {len(events)} runtime log events for session {session_id}")
    return spans, events


def resolve_evaluators(control, names):
    """Map each requested name or ID to (evaluator ID, level)."""
    known = {}
    for page in control.get_paginator("list_evaluators").paginate():
        for evaluator in page["evaluators"]:
            known[evaluator["evaluatorId"]] = evaluator
            known.setdefault(evaluator["evaluatorName"], evaluator)
    resolved = []
    for name in names:
        if name not in known:
            raise SystemExit(f"Unknown evaluator: {name}")
        resolved.append((known[name]["evaluatorId"], known[name]["level"]))
    return resolved


def targets_for(level, spans):
    """Batches of evaluationTarget values for an evaluator level."""
    if level == "SESSION":
        return [None]
    if level == "TRACE":
        ids = list(dict.fromkeys(span["traceId"] for span in spans if span.get("traceId")))
        key = "traceIds"
    else:  # TOOL_CALL: the tool execution spans
        ids = list(dict.fromkeys(
            span["spanId"] for span in spans
            if (span.get("attributes") or {}).get("gen_ai.operation.name") == "execute_tool"
        ))
        key = "spanIds"
    return [{key: ids[i:i + MAX_TARGETS]} for i in range(0, len(ids), MAX_TARGETS)]


def evaluate(client, evaluator_id, level, session_spans, spans):
    results = []
    for target in targets_for(level, spans):
        request = {"evaluatorId": evaluator_id, "evaluationInput": {"sessionSpans": session_spans}}
        if target:
            request["evaluationTarget"] = target
        results.extend(client.evaluate(**request)["evaluationResults"])
    return results


def one_line(text):
    """Judge explanations often span several lines; keep them on one."""
    return " ".join((text or "").split())


def report(session_id, results_by_evaluator):
    lines = [f"# Evaluation results for session {session_id}", ""]
    lines += ["| Evaluator | Scored | Mean score | Errors |", "|---|---|---|---|"]
    for evaluator_id, results in results_by_evaluator.items():
        values = [r["value"] for r in results if "value" in r]
        mean = f"{statistics.mean(values):.2f}" if values else "-"
        lines.append(f"| {evaluator_id} | {len(values)} | {mean} | {len(results) - len(values)} |")
    for evaluator_id, results in results_by_evaluator.items():
        lines += ["", f"## {evaluator_id}", ""]
        for result in results:
            context = result.get("context", {}).get("spanContext", {})
            where = context.get("spanId") or context.get("traceId") or context.get("sessionId")
            if "value" in result:
                lines.append(f"- `{where}` **{result.get('label')} ({result['value']})**: {one_line(result.get('explanation'))}")
            else:
                lines.append(f"- `{where}` **ERROR {result.get('errorCode')}**: {result.get('errorMessage')}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime-id", required=True, help="terraform output runtime_id")
    parser.add_argument("--session-id", required=True, help="Session ID printed by scripts/test_agent.py")
    parser.add_argument("--evaluators", required=True, help="Comma-separated evaluator IDs or names")
    parser.add_argument("--region", default="eu-central-1")
    parser.add_argument("--endpoint", default="control", help="Runtime endpoint the session ran on")
    parser.add_argument("--lookback-minutes", type=int, default=180)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    control = boto3.client("bedrock-agentcore-control", region_name=args.region)
    client = boto3.client("bedrock-agentcore", region_name=args.region)
    evaluators = resolve_evaluators(control, [name.strip() for name in args.evaluators.split(",") if name.strip()])

    spans, events = download_session_spans(
        args.region, args.runtime_id, args.endpoint, args.session_id, args.lookback_minutes
    )
    if not spans:
        raise SystemExit("No spans found. Wait a few minutes after invoking the agent and try again.")
    session_spans = spans + events

    results_by_evaluator = {}
    for evaluator_id, level in evaluators:
        print(f"\n{evaluator_id} ({level})")
        results = evaluate(client, evaluator_id, level, session_spans, spans)
        results_by_evaluator[evaluator_id] = results
        for result in results:
            if "value" in result:
                print(f"  {result.get('label')} ({result['value']}): {one_line(result.get('explanation'))[:200]}")
            else:
                print(f"  ERROR {result.get('errorCode')}: {result.get('errorMessage')}")

    out_dir = Path(args.output_dir) / args.session_id / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session-spans.json").write_text(json.dumps(session_spans, indent=2))
    (out_dir / "EvaluationResults.md").write_text(report(args.session_id, results_by_evaluator))
    print(f"\nSaved {out_dir / 'EvaluationResults.md'}")


if __name__ == "__main__":
    main()
