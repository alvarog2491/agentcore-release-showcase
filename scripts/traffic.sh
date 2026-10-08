#!/usr/bin/env bash
# Sends customer-support questions through the AgentCore Gateway so the A/B
# test has sessions to score. Each request is a new session; the gateway splits
# sessions between the control and treatment targets.
#
# Usage: scripts/traffic.sh <gateway-id> <region> <duration-seconds> [interval-seconds] [workers]
# Each worker sends one request at a time and waits interval-seconds between
# requests; more workers give the A/B test more sessions to score.
# Needs AWS credentials in the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# optional AWS_SESSION_TOKEN) with bedrock-agentcore:InvokeGateway on the gateway.
set -euo pipefail

gateway_id=$1
region=$2
duration=$3
interval=${4:-15}
workers=${5:-1}
target=${CONTROL_TARGET:-control}
url="https://${gateway_id}.gateway.bedrock-agentcore.${region}.amazonaws.com/${target}/invocations"

# Returns-heavy on purpose: most of the support workflow's rules apply to returns.
prompts=(
  "Hi, I'm ana@example.com. Where is my keyboard?"
  "Can I return the headphones from order ORD-1001?"
  "Please open a return for the USB-C cable in ORD-1001, it doesn't charge my phone."
  "I'm ben@example.com, I want to return my monitor."
  "What's the tracking number for ORD-1002?"
  "Please return the gift card in ORD-1004, I don't need it."
  "When will ORD-1002 be delivered?"
  "I'd like to send back the mouse from ORD-1004."
  "Has order ORD-1004 shipped yet?"
  "Return the headphones from ORD-1001 please, they hurt my ears."
  "I'm ana@example.com, which of my orders are still on their way?"
  "When was ORD-1001 delivered?"
)

end=$((SECONDS + duration))

# Worker $1 sends prompts round robin, starting at its own offset.
send_loop() {
  local worker=$1 sent=0 failed=0 status prompt session_id body
  local response
  response=$(mktemp)
  while ((SECONDS < end)); do
    prompt=${prompts[$(((sent * workers + worker) % ${#prompts[@]}))]}
    session_id=$(uuidgen | tr '[:upper:]' '[:lower:]')
    body=$(jq -cn --arg p "$prompt" '{prompt: $p}')
    status=$(curl -sS -o "$response" -w '%{http_code}' --max-time 120 \
      --aws-sigv4 "aws:amz:${region}:bedrock-agentcore" \
      --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
      ${AWS_SESSION_TOKEN:+-H "x-amz-security-token: ${AWS_SESSION_TOKEN}"} \
      -H "Content-Type: application/json" \
      -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: ${session_id}" \
      -d "$body" -X POST "$url") || status=000
    sent=$((sent + 1))
    if [[ $status != 200 ]]; then
      failed=$((failed + 1))
      echo "[w${worker} ${sent}] HTTP $status: $(head -c 300 "$response" 2>/dev/null)"
    else
      echo "[w${worker} ${sent}] $prompt -> $(jq -r '.result // .' "$response" | tr '\n' ' ' | head -c 160)"
    fi
    sleep "$interval"
  done
  rm -f "$response"
  echo "Worker $worker sent $sent requests, $failed failed."
}

for ((w = 0; w < workers; w++)); do
  send_loop "$w" &
done
wait
