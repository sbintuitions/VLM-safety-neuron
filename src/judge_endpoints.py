"""Azure OpenAI endpoints for the GPT-4o judge, read from the environment.

Single endpoint:
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    export AZURE_OPENAI_API_KEY=<key>

Several endpoints (requests are spread round-robin across them):
    export AZURE_OPENAI_ENDPOINTS=https://<a>.openai.azure.com/,https://<b>.openai.azure.com/
    export AZURE_OPENAI_API_KEYS=<key-a>,<key-b>      # or one shared AZURE_OPENAI_API_KEY

The judge calls an Azure deployment named "gpt-4o" (override with --judge-deployment).
"""

import os
import sys
from typing import Dict, List

API_VERSION = "2024-12-01-preview"


def _split(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def load_judge_endpoints() -> List[Dict[str, str]]:
    """Return [{"name", "endpoint", "key"}, ...]; empty if nothing is configured."""
    endpoints = _split(os.environ.get("AZURE_OPENAI_ENDPOINTS", ""))
    if not endpoints and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        endpoints = [os.environ["AZURE_OPENAI_ENDPOINT"].strip()]
    keys = _split(os.environ.get("AZURE_OPENAI_API_KEYS", ""))
    if not keys and os.environ.get("AZURE_OPENAI_API_KEY"):
        keys = [os.environ["AZURE_OPENAI_API_KEY"].strip()] * len(endpoints)
    if endpoints and len(keys) != len(endpoints):
        raise ValueError(
            f"got {len(keys)} API keys for {len(endpoints)} judge endpoints: set AZURE_OPENAI_API_KEY "
            "(one key for every endpoint) or AZURE_OPENAI_API_KEYS (one key per entry of "
            "AZURE_OPENAI_ENDPOINTS)"
        )
    return [{"name": f"endpoint{i}", "endpoint": ep, "key": key}
            for i, (ep, key) in enumerate(zip(endpoints, keys))]


def require_judge_endpoints() -> List[Dict[str, str]]:
    """Configured endpoints; exits with a clear message if there are none."""
    eps = load_judge_endpoints()
    if not eps:
        sys.exit("The LLM judge needs Azure OpenAI credentials: set AZURE_OPENAI_ENDPOINT and "
                 "AZURE_OPENAI_API_KEY (or the plural forms), see src/judge_endpoints.py.")
    return eps
