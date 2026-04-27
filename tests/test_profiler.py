"""Tests for data profiler."""

from __future__ import annotations

from pydantic_ai_cloudflare.data_profiler import profile_data

SAMPLE_DATA = [
    {
        "account_id": "ACC001",
        "company": "NexaTech",
        "industry": "SaaS",
        "revenue": "$180M",
        "employees": 1200,
        "region": "US",
        "tech_stack": "AWS, Kubernetes, Go",
        "description": (
            "NexaTech provides cloud-native infrastructure monitoring "
            "and observability tools for DevOps teams. They recently "
            "launched AI-powered anomaly detection."
        ),
        "founded": "2018-03-15",
    },
    {
        "account_id": "ACC002",
        "company": "DataPulse",
        "industry": "Analytics",
        "revenue": "$45M",
        "employees": 300,
        "region": "EU",
        "tech_stack": "GCP, Python, TensorFlow",
        "description": (
            "DataPulse offers real-time analytics for IoT devices, "
            "processing millions of events per second with their "
            "proprietary stream processing engine."
        ),
        "founded": "2020-07-01",
    },
    {
        "account_id": "ACC003",
        "company": "CloudGuard",
        "industry": "Security",
        "revenue": "$90M",
        "employees": 600,
        "region": "US",
        "tech_stack": "Azure, Rust, Zero Trust",
        "description": (
            "CloudGuard specializes in zero trust network access "
            "and cloud security posture management for enterprises."
        ),
        "founded": "2019-11-20",
    },
]


class TestProfileData:
    def test_auto_detects_id(self) -> None:
        dd = profile_data(SAMPLE_DATA)
        assert dd.id_column == "account_id"

    def test_explicit_id(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="company")
        assert dd.id_column == "company"

    def test_detects_text(self) -> None:
        # threshold=60 because sample descriptions are ~120 chars
        dd = profile_data(SAMPLE_DATA, id_column="account_id", text_threshold=60)
        assert "description" in dd.text_columns

    def test_detects_categorical(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        cats = dd.categorical_columns
        assert "industry" in cats
        assert "region" in cats

    def test_detects_numeric(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        assert "employees" in dd.numeric_columns

    def test_detects_list(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        assert "tech_stack" in dd.list_columns

    def test_detects_date(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        assert "founded" in dd.date_columns

    def test_set_type_override(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        dd.set_type("company", "skip")
        assert dd.columns["company"].role == "skip"

    def test_summary(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        s = dd.summary()
        assert "account_id" in s
        assert "id" in s
        assert "description" in s

    def test_to_dict(self) -> None:
        dd = profile_data(SAMPLE_DATA, id_column="account_id")
        d = dd.to_dict()
        assert d["id_column"] == "account_id"
        assert "description" in d["columns"]

    def test_empty_data(self) -> None:
        dd = profile_data([], id_column="id")
        assert len(dd.columns) == 0

    def test_sparse_column_skipped(self) -> None:
        data = [
            {"id": "1", "name": "A", "notes": None},
            {"id": "2", "name": "B", "notes": None},
            {"id": "3", "name": "C", "notes": None},
        ]
        dd = profile_data(data, id_column="id")
        assert dd.columns["notes"].role == "skip"
