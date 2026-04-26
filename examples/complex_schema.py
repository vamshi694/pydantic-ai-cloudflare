"""Complex nested schema — market research report.

Demonstrates that pydantic-ai-cloudflare handles deeply nested
Pydantic models with lists, dicts, optionals, and 7+ nested types.

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/complex_schema.py
"""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_cloudflare import BrowserRunToolset, cloudflare_model


class FundingRound(BaseModel):
    round_type: str  # Seed, Series A, etc.
    amount_usd: int | None = None
    date: str | None = None
    lead_investor: str | None = None


class Executive(BaseModel):
    name: str
    title: str


class Competitor(BaseModel):
    name: str
    overlap: str  # direct, indirect, adjacent
    strength: str  # stronger, weaker, comparable


class ProductFeature(BaseModel):
    name: str
    category: str
    is_differentiator: bool


class PricingTier(BaseModel):
    name: str
    price: str
    billing: str
    features: list[str]
    limits: dict[str, str]


class MarketSegment(BaseModel):
    name: str
    tam_usd: str | None = None
    trends: list[str]


class CompanyReport(BaseModel):
    """Full company research report — 20+ fields, 7 nested models."""

    # Basic
    name: str
    website: str
    founded_year: int
    headquarters: str
    employee_count: int
    description: str
    business_model: str

    # People
    founders: list[Executive]
    leadership: list[Executive]

    # Product
    products: list[ProductFeature]
    pricing: list[PricingTier]

    # Market
    target_markets: list[MarketSegment]
    competitors: list[Competitor]

    # Funding
    funding_rounds: list[FundingRound]
    total_funding_usd: int | None = None
    is_public: bool

    # SWOT
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]


agent = Agent(
    cloudflare_model(),
    output_type=CompanyReport,
    toolsets=[BrowserRunToolset(tools=["browse", "extract"])],
    system_prompt=(
        "You are a market research analyst. When asked about a company, "
        "use the browse and extract tools to gather real data from their "
        "website. Return a comprehensive company report."
    ),
)

result = agent.run_sync(
    "Research Cloudflare from their website (cloudflare.com). "
    "Fill in as much of the report as you can from public information."
)

report = result.output  # CompanyReport — validated, typed, complete

print(f"Company: {report.name}")
print(f"Founded: {report.founded_year}")
print(f"Employees: {report.employee_count}")
print(f"Business model: {report.business_model}")
print("\nFounders:")
for f in report.founders:
    print(f"  {f.name} — {f.title}")
print(f"\nProducts ({len(report.products)}):")
for p in report.products[:5]:
    print(f"  {'*' if p.is_differentiator else '-'} {p.name} ({p.category})")
print(f"\nPricing ({len(report.pricing)} tiers):")
for t in report.pricing:
    print(f"  {t.name}: {t.price}")
print("\nSWOT:")
print(f"  Strengths: {report.strengths[:3]}")
print(f"  Weaknesses: {report.weaknesses[:3]}")
