"""Data profiler — auto-infer column types and build a data dictionary.

Takes a dataset (list of dicts, CSV path, or DataFrame-like) and uses
heuristics + LLM to classify each column:

  - id: unique identifier (used as entity node ID)
  - text: long text / paragraphs (embedded, entity-extracted, summarized)
  - categorical: low-cardinality strings (become feature nodes)
  - numeric: numbers (bucketed into ranges for graph edges)
  - list: comma/semicolon-separated values (split into individual nodes)
  - date: timestamps (bucketed by period)
  - skip: columns to ignore (too sparse, all nulls, etc.)

The profiler returns a DataDictionary that the user can review and edit
before passing to KnowledgeGraph.build_from_records().

    from pydantic_ai_cloudflare import profile_data

    # Auto-profile
    dd = profile_data(records, id_column="account_id")
    print(dd)  # shows each column with inferred type + reasoning

    # Edit if needed
    dd.set_type("notes", "text")  # override a misclassified column
    dd.set_type("internal_code", "skip")  # exclude a column

    # Build graph using the dictionary
    await kg.build_from_records(records, data_dict=dd)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class ColumnProfile:
    """Profile of a single column."""

    __slots__ = ("name", "dtype", "role", "reason", "sample_values", "stats")

    def __init__(
        self,
        name: str,
        dtype: str,
        role: str,
        reason: str,
        sample_values: list[Any],
        stats: dict[str, Any],
    ) -> None:
        self.name = name
        self.dtype = dtype  # python type: str, int, float, mixed
        self.role = role  # id, text, categorical, numeric, list, date, skip
        self.reason = reason
        self.sample_values = sample_values
        self.stats = stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "reason": self.reason,
            "sample_values": self.sample_values[:3],
            "stats": self.stats,
        }


class DataDictionary:
    """Data dictionary for a tabular dataset.

    Maps each column to a role (id, text, categorical, numeric, list,
    date, skip) with reasoning. Editable by the user before building
    the knowledge graph.
    """

    def __init__(self, columns: list[ColumnProfile], id_column: str) -> None:
        self.columns = {c.name: c for c in columns}
        self.id_column = id_column

    def set_type(self, column: str, role: str) -> None:
        """Override the inferred type for a column."""
        if column in self.columns:
            self.columns[column].role = role
            self.columns[column].reason = "manually set by user"

    def get_columns_by_role(self, role: str) -> list[str]:
        """Get column names matching a role."""
        return [c.name for c in self.columns.values() if c.role == role]

    @property
    def text_columns(self) -> list[str]:
        return self.get_columns_by_role("text")

    @property
    def categorical_columns(self) -> list[str]:
        return self.get_columns_by_role("categorical")

    @property
    def numeric_columns(self) -> list[str]:
        return self.get_columns_by_role("numeric")

    @property
    def list_columns(self) -> dict[str, str]:
        """Returns {column_name: edge_type}."""
        return {c: f"HAS_{c.upper()}" for c in self.get_columns_by_role("list")}

    @property
    def date_columns(self) -> list[str]:
        return self.get_columns_by_role("date")

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Data Dictionary ({len(self.columns)} columns)",
            f"ID column: {self.id_column}",
            "",
        ]
        for col in self.columns.values():
            lines.append(f"  {col.name:30s} → {col.role:12s} ({col.dtype}) | {col.reason}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def review(self) -> str:
        """Show each column with confidence flags for uncertain guesses.

        Columns that might be wrong are flagged with ⚠️ so you know
        what to check before building the graph.
        """
        lines = [
            f"Data Dictionary ({len(self.columns)} columns)",
            f"ID column: {self.id_column}",
            "",
            f"{'Column':<25s} {'Role':<14s} {'Flag':<4s} Reason",
            f"{'-' * 25} {'-' * 14} {'-' * 4} {'-' * 40}",
        ]
        for col in self.columns.values():
            flag = ""
            # Flag uncertain classifications
            if "high cardinality" in col.reason:
                flag = "⚠️"  # might be text, id, or skip
            elif col.role == "list" and col.stats.get("avg_items", 0) < 1.5:
                flag = "⚠️"  # might be text with commas, not a real list
            elif col.role == "categorical" and col.stats.get("cardinality", 0) > 30:
                flag = "⚠️"  # high cardinality categorical — maybe text?
            elif col.role == "skip":
                flag = "⏭️"

            lines.append(f"  {col.name:<25s} {col.role:<14s} {flag:<4s} {col.reason}")

        flagged = sum(
            1
            for c in self.columns.values()
            if "high cardinality" in c.reason
            or (c.role == "list" and c.stats.get("avg_items", 0) < 1.5)
            or (c.role == "categorical" and c.stats.get("cardinality", 0) > 30)
        )
        if flagged:
            lines.append(f"\n⚠️ {flagged} column(s) flagged — review with dd.set_type(col, role)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_column": self.id_column,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }


def profile_data(
    records: list[dict[str, Any]],
    *,
    id_column: str | None = None,
    sample_size: int = 100,
    text_threshold: int = 50,
    categorical_max_cardinality: int = 50,
    list_separator_pattern: str = r"[,;|]",
) -> DataDictionary:
    """Profile a dataset and infer column types.

    Uses heuristics (no LLM call) for fast profiling. For LLM-enhanced
    profiling, use profile_data_with_llm().

    Args:
        records: List of row dicts.
        id_column: If known, the ID column. Otherwise auto-detected.
        sample_size: Number of rows to sample for profiling.
        text_threshold: Avg char length above which a string column
            is classified as 'text' instead of 'categorical'.
        categorical_max_cardinality: Max unique values for 'categorical'.
            Above this, it's either 'text' or 'id'.
        list_separator_pattern: Regex for detecting list columns.

    Returns:
        A DataDictionary ready for review/editing.
    """
    if not records:
        return DataDictionary([], id_column or "")

    # Sample
    sample = records[:sample_size]
    n = len(sample)
    all_cols = list(sample[0].keys())
    profiles: list[ColumnProfile] = []

    # Detect ID column if not provided
    if id_column is None:
        id_column = _detect_id_column(sample, all_cols)

    for col in all_cols:
        values = [r.get(col) for r in sample]
        non_null = [
            v for v in values if v is not None and str(v).strip() not in ("", "nan", "None")
        ]
        null_rate = 1 - (len(non_null) / n) if n > 0 else 1.0

        stats: dict[str, Any] = {
            "null_rate": round(null_rate, 2),
            "count": len(non_null),
        }

        # Skip if too sparse
        if null_rate > 0.9:
            profiles.append(
                ColumnProfile(
                    col,
                    "mixed",
                    "skip",
                    f"{null_rate:.0%} null — too sparse",
                    values[:3],
                    stats,
                )
            )
            continue

        # ID column
        if col == id_column:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "id",
                    "designated ID column",
                    values[:3],
                    stats,
                )
            )
            continue

        # Detect type from values
        str_vals = [str(v) for v in non_null]
        unique = set(str_vals)
        cardinality = len(unique)
        avg_len = sum(len(s) for s in str_vals) / len(str_vals) if str_vals else 0
        stats["cardinality"] = cardinality
        stats["avg_length"] = round(avg_len, 1)

        # Numeric check
        numeric_count = sum(1 for v in non_null if _is_numeric(v))
        if numeric_count / max(len(non_null), 1) > 0.8:
            profiles.append(
                ColumnProfile(
                    col,
                    "numeric",
                    "numeric",
                    f"{numeric_count}/{len(non_null)} values are numeric, avg={avg_len:.0f}",
                    values[:3],
                    stats,
                )
            )
            continue

        # Skip patterns: URLs, phones, UUIDs (not useful as graph nodes)
        url_count = sum(1 for s in str_vals if _looks_like_url(s))
        if url_count / max(len(str_vals), 1) > 0.5:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "skip",
                    f"{url_count}/{len(str_vals)} look like URLs",
                    values[:3],
                    stats,
                )
            )
            continue

        phone_count = sum(1 for s in str_vals if _looks_like_phone(s))
        if phone_count / max(len(str_vals), 1) > 0.5:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "skip",
                    f"{phone_count}/{len(str_vals)} look like phone numbers",
                    values[:3],
                    stats,
                )
            )
            continue

        uuid_count = sum(1 for s in str_vals if _looks_like_uuid(s))
        if uuid_count / max(len(str_vals), 1) > 0.5:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "skip",
                    f"{uuid_count}/{len(str_vals)} look like UUIDs",
                    values[:3],
                    stats,
                )
            )
            continue

        # Date check
        date_count = sum(1 for s in str_vals if _looks_like_date(s))
        if date_count / max(len(str_vals), 1) > 0.6:
            profiles.append(
                ColumnProfile(
                    col,
                    "date",
                    "date",
                    f"{date_count}/{len(str_vals)} look like dates",
                    values[:3],
                    stats,
                )
            )
            continue

        # Text check first (long text naturally contains commas)
        if avg_len > text_threshold:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "text",
                    f"avg length {avg_len:.0f} chars, {cardinality} unique — long text",
                    values[:3],
                    stats,
                )
            )
            continue

        # List check (short strings with separators)
        list_count = sum(
            1 for s in str_vals if re.search(list_separator_pattern, s) and len(s.split(",")) > 1
        )
        if list_count / max(len(str_vals), 1) > 0.3:
            avg_items = sum(len(s.split(",")) for s in str_vals) / len(str_vals)
            stats["avg_items"] = round(avg_items, 1)
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "list",
                    f"{list_count}/{len(str_vals)} comma-sep (avg {avg_items:.1f} items)",
                    values[:3],
                    stats,
                )
            )
            continue

        # Categorical
        if cardinality <= categorical_max_cardinality:
            # Show top values
            top = Counter(str_vals).most_common(5)
            stats["top_values"] = top
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "categorical",
                    f"{cardinality} unique values — low cardinality",
                    values[:3],
                    stats,
                )
            )
        elif cardinality > categorical_max_cardinality and cardinality / n > 0.9:
            # High cardinality unique strings — likely an ID or name
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "categorical",
                    f"{cardinality} unique, high cardinality — may want to set as 'text' or 'skip'",
                    values[:3],
                    stats,
                )
            )
        else:
            profiles.append(
                ColumnProfile(
                    col,
                    "str",
                    "categorical",
                    f"{cardinality} unique values",
                    values[:3],
                    stats,
                )
            )

    return DataDictionary(profiles, id_column)


async def profile_data_with_llm(
    records: list[dict[str, Any]],
    *,
    id_column: str | None = None,
    model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    account_id: str | None = None,
    api_key: str | None = None,
) -> DataDictionary:
    """Profile data using LLM for smarter type inference.

    Uses heuristic profiling first, then asks the LLM to review
    and suggest corrections based on column names, sample values,
    and statistics.
    """
    from pydantic import BaseModel

    from .structured import cf_structured

    # Start with heuristic profiling
    dd = profile_data(records, id_column=id_column)

    # Build a summary for the LLM
    col_summaries = []
    for col in dd.columns.values():
        s = (
            f"Column '{col.name}': inferred as '{col.role}' ({col.dtype}). "
            f"Reason: {col.reason}. "
            f"Samples: {col.sample_values[:2]}"
        )
        col_summaries.append(s)

    class ColumnOverride(BaseModel):
        column: str
        new_role: str
        reasoning: str

    class ProfileReview(BaseModel):
        overrides: list[ColumnOverride]
        dataset_description: str

    try:
        review = await cf_structured(
            "Review this data dictionary and suggest corrections.\n\n"
            "Columns:\n" + "\n".join(col_summaries) + "\n\n"
            "Valid roles: id, text, categorical, numeric, list, date, skip.\n"
            "Only suggest overrides where the heuristic got it wrong.\n"
            "Also describe what this dataset appears to be about.",
            ProfileReview,
            model=model,
            account_id=account_id,
            api_key=api_key,
            max_tokens=2048,
        )

        for override in review.overrides:
            if override.column in dd.columns:
                dd.set_type(override.column, override.new_role)
                dd.columns[override.column].reason = f"LLM: {override.reasoning}"

    except Exception as e:
        logger.warning(f"LLM profiling failed, using heuristics only: {e}")

    return dd


# -- Heuristic helpers --


def _detect_id_column(sample: list[dict], cols: list[str]) -> str:
    """Guess which column is the ID based on name + uniqueness."""
    n = len(sample)
    # Check common ID column names
    for pattern in ["id", "account_id", "entity_id", "record_id", "uuid", "key"]:
        for col in cols:
            if col.lower() == pattern or col.lower().endswith("_id"):
                values = [r.get(col) for r in sample]
                unique = len(set(str(v) for v in values if v is not None))
                if unique / max(n, 1) > 0.9:
                    return col

    # Fallback: first column with near-unique values
    for col in cols:
        values = [r.get(col) for r in sample]
        unique = len(set(str(v) for v in values if v is not None))
        if unique / max(n, 1) > 0.95:
            return col

    return cols[0]


def _is_numeric(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    try:
        s = str(v).replace("$", "").replace(",", "").replace("%", "").strip()
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_date(s: str) -> bool:
    patterns = [
        r"\d{4}-\d{2}-\d{2}",  # 2024-01-15
        r"\d{2}/\d{2}/\d{4}",  # 01/15/2024
        r"\d{4}-\d{2}",  # 2024-01
        r"[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}",  # Jan 15, 2024
    ]
    return any(re.match(p, s.strip()) for p in patterns)


def _looks_like_url(s: str) -> bool:
    return bool(re.match(r"https?://", s.strip(), re.IGNORECASE))


def _looks_like_phone(s: str) -> bool:
    # Must start with + or ( to distinguish from dates like 2024-03-15
    return bool(re.match(r"[\+\(][\d\-\(\)\s]{6,15}$", s.strip()))


def _looks_like_uuid(s: str) -> bool:
    return bool(
        re.match(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
            s.strip(),
            re.IGNORECASE,
        )
    )
