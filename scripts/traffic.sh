#!/usr/bin/env bash
# Sends customer-support questions through the AgentCore Gateway so the A/B
# test has sessions to score. Each request is a new session; the gateway splits
# sessions between the control and treatment targets.
#
# Usage: scripts/traffic.sh <gateway-id> <region> <duration-seconds> [interval-seconds]
# Needs AWS credentials in the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# optional AWS_SESSION_TOKEN) with bedrock-agentcore:InvokeGateway on the gateway.
set -euo pipefail

gateway_id=$1
region=$2
duration=$3
interval=${4:-15}
target=${CONTROL_TARGET:-control}
url="https://${gateway_id}.gateway.bedrock-agentcore.${region}.amazonaws.com/${target}/invocations"

prompts=(
  "Hi, I'm ana@example.com. Where is my keyboard?"
  "Can I return the headphones from order ORD-1001?"
  "What did I buy in order ORD-1001 and how much was it?"
  "I'm ben@example.com, I want to return my monitor."
  "Has order ORD-1004 shipped yet?"
  "Please open a return for the USB-C cable in ORD-1001, it doesn't charge my phone."
  "What's the tracking number for ORD-1002?"
  "Can I return the gift card in ORD-1004?"
  "I'm ana@example.com, which of my orders are still on their way?"
  "When will ORD-1002 be delivered?"
)

end=$((SECONDS + duration))
sent=0
failed=0
while ((SECONDS < end)); do
  prompt=${prompts[$((sent % ${#prompts[@]}))]}
  session_id=$(uuidgen | tr '[:upper:]' '[:lower:]')
  body=$(jq -cn --arg p "$prompt" '{prompt: $p}')
  status=$(curl -sS -o /tmp/traffic-response.json -w '%{http_code}' --max-time 120 \
    --aws-sigv4 "aws:amz:${region}:bedrock-agentcore" \
    --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
    ${AWS_SESSION_TOKEN:+-H "x-amz-security-token: ${AWS_SESSION_TOKEN}"} \
    -H "Content-Type: application/json" \
    -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: ${session_id}" \
    -d "$body" -X POST "$url") || status=000
  sent=$((sent + 1))
  if [[ $status != 200 ]]; then
    failed=$((failed + 1))
    echo "[$sent] HTTP $status: $(head -c 300 /tmp/traffic-response.json 2>/dev/null)"
  else
    echo "[$sent] $prompt -> $(jq -r '.result // .' /tmp/traffic-response.json | tr '\n' ' ' | head -c 160)"
  fi
  sleep "$interval"
done
echo "Sent $sent requests, $failed failed."
