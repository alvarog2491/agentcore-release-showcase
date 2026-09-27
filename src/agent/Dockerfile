FROM public.ecr.aws/docker/library/python:3.12-slim-trixie

RUN pip install --no-cache-dir uv

ARG UV_DEFAULT_INDEX
ARG UV_INDEX

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_INDEX=${UV_INDEX} \
    PATH="/app/.venv/bin:$PATH"

RUN useradd -m -u 1000 bedrock_agentcore

# Build context is the repo root: the uv workspace (root pyproject.toml + the
# single uv.lock) lives there. Only the agent package's dependencies are installed.
COPY pyproject.toml uv.lock ./
COPY src/agent/pyproject.toml src/agent/README.md src/agent/
RUN uv sync --frozen --no-dev --package myagent --no-install-workspace

COPY --chown=bedrock_agentcore:bedrock_agentcore src/agent/ src/agent/
RUN uv sync --frozen --no-dev --package myagent

WORKDIR /app/src/agent
USER bedrock_agentcore

# AgentCore Runtime service contract ports
# https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html
# 8080: HTTP Mode
# 8000: MCP Mode
# 9000: A2A Mode
EXPOSE 8080 8000 9000

CMD ["opentelemetry-instrument", "python", "-m", "main"]
