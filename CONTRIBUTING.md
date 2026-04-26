# Contributing

Thanks for wanting to contribute!

## Development Setup

```bash
git clone https://github.com/vamshi694/pydantic-ai-cloudflare.git
cd pydantic-ai-cloudflare
uv sync --extra dev
```

## Running Tests

```bash
uv run pytest tests/ -v
```

All tests use mocked HTTP -- no Cloudflare account needed for unit tests.

For integration tests, set:
```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-api-token"
```

## Code Style

- We use `ruff` for linting and formatting
- Run `uv run ruff check --fix src/ tests/` before committing
- Run `uv run ruff format src/ tests/`

## Adding a New Cloudflare Service Integration

1. Create a module in `src/pydantic_ai_cloudflare/`
2. Add tests in `tests/` with mocked HTTP calls
3. Export from `__init__.py`
4. Add an example in `examples/`
5. Update README

## Pull Requests

- Open an issue first for larger changes
- Keep PRs focused -- one feature per PR
- Include tests for new functionality
