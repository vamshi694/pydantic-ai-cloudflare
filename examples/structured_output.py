"""Structured output with Pydantic models.

The LLM output is validated against a Pydantic schema automatically.
No string parsing, no JSON loading -- you get typed Python objects.

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/structured_output.py
"""

from pydantic import BaseModel
from pydantic_ai import Agent


class CompanyProfile(BaseModel):
    """Structured company information."""

    name: str
    industry: str
    founded_year: int
    headquarters: str
    key_products: list[str]
    one_sentence_description: str


agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    output_type=CompanyProfile,
    system_prompt="You are a business analyst. Provide accurate company information.",
)

result = agent.run_sync("Tell me about Cloudflare")
company = result.output

print(f"Company: {company.name}")
print(f"Industry: {company.industry}")
print(f"Founded: {company.founded_year}")
print(f"HQ: {company.headquarters}")
print(f"Products: {', '.join(company.key_products)}")
print(f"Description: {company.one_sentence_description}")
