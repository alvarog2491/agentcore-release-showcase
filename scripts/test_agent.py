"""Invoke the deployed agent with multi-turn customer sessions to evaluate.

Each category is one session of five turns that build on each other, so the
evaluators see follow-up questions, not only isolated prompts. The session IDs
printed at the end are the input of scripts/evaluate.py.

Usage:
    uv run scripts/test_agent.py --runtime-arn <runtime-arn> [--category returns ...]
"""

import argparse
import json
import uuid

import boto3

SESSIONS = {
    "orders": [
        "Hi, I'm ana@example.com. Which orders do I have?",
        "What did I buy in ORD-1001 and how much was it in total?",
        "And what is in ORD-1004?",
        "Which of my orders are still on their way?",
        "Can you also list the orders of ben@example.com?",
    ],
    "shipping": [
        "Where is my keyboard? My email is ana@example.com.",
        "What's the tracking number?",
        "When will it arrive?",
        "Has ORD-1004 shipped yet?",
        "When was ORD-1001 delivered?",
    ],
    "returns": [
        "I'm ana@example.com. Can I return the headphones from ORD-1001?",
        "Please open a return for the USB-C cable in ORD-1001, it doesn't charge my phone.",
        "Can I return the gift card in ORD-1004?",
        "What about the mouse in ORD-1004?",
        "My friend ben@example.com wants to return his monitor. Can he?",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime-arn", required=True, help="terraform output runtime_arn")
    parser.add_argument("--region", default="eu-central-1")
    parser.add_argument("--qualifier", default="control", help="Runtime endpoint to call (default: control)")
    parser.add_argument("--category", nargs="+", choices=sorted(SESSIONS), default=list(SESSIONS))
    args = parser.parse_args()

    client = boto3.client("bedrock-agentcore", region_name=args.region)
    session_ids = {}

    for category in args.category:
        # Runtime session IDs must be at least 33 characters long.
        session_id = f"{category}-{uuid.uuid4()}"
        session_ids[category] = session_id
        print(f"\n=== {category} (session {session_id}) ===")

        for turn, prompt in enumerate(SESSIONS[category], start=1):
            response = client.invoke_agent_runtime(
                agentRuntimeArn=args.runtime_arn,
                qualifier=args.qualifier,
                runtimeSessionId=session_id,
                payload=json.dumps({"prompt": prompt}),
            )
            answer = json.loads(response["response"].read()).get("result", "")
            print(f"\n[{turn}] Customer: {prompt}\n    Agent: {answer}")

    print("\nSession IDs:")
    for category, session_id in session_ids.items():
        print(f"  {category}: {session_id}")


if __name__ == "__main__":
    main()
