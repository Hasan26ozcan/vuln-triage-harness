# Docker sandbox for Stage 6 exec-based evaluation.
#
# Each patch test runs in an isolated container: the vulnerable code and
# the CWE-specific test are mounted as files, the patch is applied, and
# the test runs via `pytest`.  The container has a read-only root filesystem
# and a writable /tmp mount — the patched code cannot touch the host.
#
# Build:  docker build -t vuln-triage-sandbox:python3.11 -f sandbox/Dockerfile .
# Run:    docker run --rm -v /host/tmp/code:/code:ro -v /host/tmp/test:/test:ro vuln-triage-sandbox:python3.11

# ---- Base image: slim Python with minimal attack surface ----
FROM python:3.11-slim

# Avoid interactive prompts during apt installs.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only what we need to run a Python test file.
# No compilers, no package managers beyond pip — minimal footprint.
RUN pip install --no-input pytest==8.3.5

# Non-root user for the container.
RUN useradd --uid 1000 --create-home sandbox
WORKDIR /sandbox
USER sandbox

# ---- Runtime: read-only filesystem for safety ----
# The container runs with --read-only and a tmpfs at /tmp.
# Code and test files are mounted read-only at /code and /test.
# pytest output goes to stdout, which the parent process captures.
