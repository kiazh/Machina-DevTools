"""Tests for machina.core.metrics."""

import json
import logging

from machina.core.metrics import JsonFormatter, MetricsCollector, metrics, timed


def test_latency_stats():
    m = MetricsCollector()
    for v in (10.0, 20.0, 30.0):
        m.record_latency("op", v)
    stats = m.get_latency_stats("op")
    assert (stats["count"], stats["min"], stats["max"], stats["mean"]) == (3, 10.0, 30.0, 20.0)


def test_counters_and_error_rate():
    m = MetricsCollector()
    m.increment_counter("op")
    m.increment_counter("op")
    m.record_error("op")
    assert m.get_counter("op") == 2
    assert m.get_error_count("op") == 1
    assert m.get_error_rate("op") == 0.5


def test_token_usage():
    m = MetricsCollector()
    m.record_tokens("s1", 100, 50)
    m.record_tokens("s1", 200, 100)
    # Only cumulative process totals are retained (the per-session breakdown
    # had no production reader; Prometheus scrapes token_totals).
    assert m.get_all_metrics()["token_totals"] == {
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "total_tokens": 450,
    }
    assert "token_usage" not in m.get_all_metrics()


def test_db_and_llm_calls():
    m = MetricsCollector()
    m.record_db_call("execute", 5.0)
    m.record_db_call("fetch", 10.0, is_error=True)
    m.record_llm_call("s1", "gpt-4", 500.0, 100, 50)
    assert m.get_latency_stats("db.execute")["count"] == 1
    assert m.get_error_count("db.fetch") == 1
    assert m.get_latency_stats("llm.gpt-4")["count"] == 1


def test_all_metrics_and_reset():
    m = MetricsCollector()
    m.increment_counter("op")
    assert {
        "latencies",
        "counters",
        "errors",
        "error_rates",
        "token_totals",
    } <= m.get_all_metrics().keys()
    m.reset()
    assert m.get_counter("op") == 0


def test_timed_records_to_global_collector():
    before = metrics.get_counter("timed.test.calls")
    with timed("timed.test"):
        pass
    assert metrics.get_counter("timed.test.calls") == before + 1


def test_json_formatter():
    record = logging.LogRecord("t", logging.INFO, "t.py", 1, "hello", (), None)
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "hello" and data["level"] == "INFO"


def test_metric_series_are_bounded_into_other():
    m = MetricsCollector()
    m._MAX_SERIES = 2
    for index in range(5):
        m.record_latency(f"op{index}", 1.0)
        m.increment_counter(f"op{index}")
        m.record_error(f"op{index}")
    assert len(m._latencies) == 3
    assert len(m._counters) == 3
    assert len(m._errors) == 3
    assert m.get_counter("other") == 3
    assert m.get_error_count("other") == 3
    assert m.get_latency_stats("other")["count"] == 3
