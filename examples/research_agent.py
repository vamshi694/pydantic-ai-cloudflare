"""Research agent that browses the web and extracts structured data.

Combines Browser Run (web interaction) + Workers AI (inference) +
Pydantic (structured output) to build a research agent that can
analyze any website and return validated Python objects.

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/research_agent.py
"""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_cloudflare import BrowserRunToolset


class PricingPlan(BaseModel):
    name: str
    price: str
    features: list[str]


class PricingAnalysis(BaseModel):
    company: str
    plans: list[PricingPlan]
    cheapest_paid_plan: str
    has_free_tier: bool
    summary: str


agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    output_type=PricingAnalysis,
    toolsets=[BrowserRunToolset(tools=["browse", "extract"])],
    system_prompt=(
        "You are a pricing analyst. When asked about a company's pricing, "
        "use the browse or extract tools to visit their pricing page and "
        "return structured pricing information."
    ),
)

result = agent.run_sync("Analyze Cloudflare's pricing from https://www.cloudflare.com/plans/")

analysis = result.output
print(f"Company: {analysis.company}")
print(f"Free tier: {'Yes' if analysis.has_free_tier else 'No'}")
print(f"Cheapest paid: {analysis.cheapest_paid_plan}")
print(f"\nPlans:")
for plan in analysis.plans:
    print(f"  {plan.name}: {plan.price}")
    for feature in plan.features[:3]:
        print(f"    - {feature}")
print(f"\nSummary: {analysis.summary}")
