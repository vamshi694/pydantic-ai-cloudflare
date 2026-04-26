"""Complex nested schema — company research report.

Demonstrates cf_structured() handling deeply nested Pydantic models
with Literal types, lists of nested models, and Optional fields.

Tested on all 6 major Workers AI models: Llama, Qwen, Kimi, Gemma, GLM, DeepSeek.

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/complex_schema.py
"""

from typing import Literal

from pydantic import BaseModel

from pydantic_ai_cloudflare import cf_structured_sync


class NewsItem(BaseModel):
    headline: str
    summary: str
    relevance: str


class TriggerEvent(BaseModel):
    event: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]


class HiringSignal(BaseModel):
    role: str
    insight: str


class Company(BaseModel):
    name: str
    industry: str
    employees: str


class MarketIntel(BaseModel):
    news: list[NewsItem]
    triggers: list[TriggerEvent]
    hiring: list[HiringSignal]


class Positioning(BaseModel):
    value_prop: str
    strengths: list[str]
    risks: list[str]
    recommended_actions: list[str]


class TechAssessment(BaseModel):
    score: int
    signals: list[str]
    assessment: str


class NextStep(BaseModel):
    action: str
    reasoning: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]


class CompanyReport(BaseModel):
    """7 top-level fields, 7 nested models, Literal types."""

    overview: str
    tldr: str
    company: Company
    market: MarketIntel
    positioning: Positioning
    tech: TechAssessment
    next_steps: list[NextStep]


# Using cf_structured_sync() which handles Workers AI quirks:
# - Schema injection into system prompt
# - response_format: json_object
# - Custom retry with error feedback
# - Works on ALL Workers AI models
result = cf_structured_sync(
    "Research report on a fictional SaaS company called 'NovaPay' that provides "
    "payment processing for e-commerce. They have 500 employees, are growing fast, "
    "and recently raised Series C. Include 3 news items, 3 triggers, 2 hiring "
    "signals, and 5 next steps.",
    CompanyReport,
    model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)

print(f"Company: {result.company.name} ({result.company.industry})")
print(f"Employees: {result.company.employees}")
print(f"\nOverview: {result.overview[:200]}...")
print(f"\nNews ({len(result.market.news)}):")
for n in result.market.news:
    print(f"  {n.headline}")
print(f"\nTriggers ({len(result.market.triggers)}):")
for t in result.market.triggers:
    print(f"  [{t.priority}] {t.event}")
print(f"\nHiring ({len(result.market.hiring)}):")
for h in result.market.hiring:
    print(f"  {h.role}: {h.insight}")
print(f"\nPositioning: {result.positioning.value_prop[:100]}")
print(f"Strengths: {result.positioning.strengths}")
print(f"\nTech score: {result.tech.score}")
print(f"\nNext steps ({len(result.next_steps)}):")
for s in result.next_steps:
    print(f"  [{s.priority}] {s.action}")
