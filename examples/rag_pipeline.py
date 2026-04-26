"""RAG pipeline: crawl docs → embed → store in Vectorize → query.

Full Cloudflare-native RAG with zero external dependencies:
  Browser Run (crawl) → Workers AI (embed) → Vectorize (store) → Workers AI (query)

Prerequisites:
    1. Create a Vectorize index:
       npx wrangler vectorize create my-docs --dimensions 768 --metric cosine

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/rag_pipeline.py
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_cloudflare import BrowserRunToolset, VectorizeToolset


class Answer(BaseModel):
    response: str
    sources: list[str]
    confidence: str


async def main() -> None:
    # Step 1: Crawl docs and store in Vectorize
    browser = BrowserRunToolset(tools=["browse"])
    vectorize = VectorizeToolset(index_name="my-docs")

    print("Step 1: Loading documentation pages...")
    pages = [
        "https://developers.cloudflare.com/browser-run/",
        "https://developers.cloudflare.com/workers-ai/",
    ]

    for url in pages:
        content = await browser._browse(url)
        await vectorize._store(content[:2000], source=url)
        print(f"  Stored: {url} ({len(content)} chars)")

    # Step 2: Query with RAG
    print("\nStep 2: Querying with RAG agent...")
    agent = Agent(
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        output_type=Answer,
        toolsets=[vectorize],
        system_prompt=(
            "Answer questions using the search_knowledge tool to find "
            "relevant information. Always cite your sources."
        ),
    )

    result = await agent.run("What is Cloudflare Browser Run and what can it do?")

    print(f"\nAnswer: {result.output.response}")
    print(f"Sources: {', '.join(result.output.sources)}")
    print(f"Confidence: {result.output.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
