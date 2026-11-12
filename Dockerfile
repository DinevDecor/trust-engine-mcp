# Runs the stdio MCP server (used by directories such as Glama to start the server and list its tools).
# No key needed. The server makes no network call until a data tool is used (GET agenttrustapi.com only).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
ENTRYPOINT ["trust-engine-mcp"]
