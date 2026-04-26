"""Persistent chat agent with D1 message history.

Conversations survive across sessions using Cloudflare D1
(serverless SQLite). Each session has its own message history.

Prerequisites:
    1. Create a D1 database:
       npx wrangler d1 create my-chat-db

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/persistent_chat.py
"""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_cloudflare import D1MessageHistory, cloudflare_model

# Replace with your actual D1 database ID
D1_DATABASE_ID = "your-d1-database-id"
SESSION_ID = "demo-session-001"


async def main() -> None:
    history = D1MessageHistory(database_id=D1_DATABASE_ID)

    agent = Agent(
        cloudflare_model(),
        system_prompt=(
            "You are a helpful assistant. Remember the context from "
            "previous messages in this conversation."
        ),
    )

    # Load any existing conversation
    messages = await history.get_messages(SESSION_ID)
    if messages:
        print(f"Loaded {len(messages)} previous messages for session {SESSION_ID}")
    else:
        print(f"Starting new session: {SESSION_ID}")

    # Have a multi-turn conversation
    questions = [
        "My name is Alice and I work at Cloudflare.",
        "What company do I work at?",
        "What's my name?",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        result = await agent.run(question, message_history=messages)
        messages = result.all_messages()
        print(f"Assistant: {result.output}")

    # Save the conversation for next time
    await history.save_messages(SESSION_ID, messages)
    print(f"\nSaved {len(messages)} messages to D1")

    # List all sessions
    sessions = await history.list_sessions()
    print(f"\nActive sessions: {len(sessions)}")
    for s in sessions:
        print(f"  {s['session_id']}: {s['message_count']} messages")


if __name__ == "__main__":
    asyncio.run(main())
