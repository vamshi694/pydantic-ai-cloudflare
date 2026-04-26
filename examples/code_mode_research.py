"""Research agent with Code Mode (Monty).

Instead of the LLM making 10+ sequential tool calls, it writes one Python
script that calls browse/extract/discover_links in parallel. Monty executes
it safely in a sandbox. ~80% fewer LLM round-trips.

Requires: pip install pydantic-ai-harness[code-mode]

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/code_mode_research.py
"""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_cloudflare import BrowserRunToolset

try:
    from pydantic_ai_harness import CodeMode

    HAS_CODE_MODE = True
except ImportError:
    HAS_CODE_MODE = False
    print("Install pydantic-ai-harness[code-mode] for Code Mode support:")
    print("  pip install 'pydantic-ai-harness[code-mode]'")


class ResearchReport(BaseModel):
    topic: str
    sources: list[str]
    findings: list[str]
    conclusion: str


if HAS_CODE_MODE:
    agent = Agent(
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        output_type=ResearchReport,
        capabilities=[CodeMode()],
        toolsets=[BrowserRunToolset(tools=["browse", "extract", "discover_links"])],
        system_prompt=(
            "You are a research analyst. Use the available tools to browse "
            "websites, extract data, and discover links. Write Python code "
            "that uses asyncio.gather for parallel operations when possible."
        ),
    )

    result = agent.run_sync(
        "Research what Cloudflare Browser Run is by visiting "
        "https://developers.cloudflare.com/browser-run/ and its sub-pages. "
        "Summarize the key features and use cases."
    )

    report = result.output
    print(f"Topic: {report.topic}")
    print(f"\nSources:")
    for src in report.sources:
        print(f"  - {src}")
    print(f"\nFindings:")
    for finding in report.findings:
        print(f"  - {finding}")
    print(f"\nConclusion: {report.conclusion}")
else:
    print("Skipping Code Mode example (pydantic-ai-harness not installed)")
