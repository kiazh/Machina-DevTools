# Benchmarks

This directory is the future home of standalone benchmark/load suites for
Machina. It is intentionally empty for now — an audit of the Machina-Harness
checkout (2026-10-15) found **no standalone benchmark suite** to migrate.

Benchmark-adjacent surfaces that exist today and stay where they are:

| Surface | Location | Why it is not migrated |
| --- | --- | --- |
| Unified offline benchmark (`python -m machina.research.benchmark`) | `machina/research/benchmark.py` | Shipped package code; writes its own JSON artifact; tested by `tests/test_benchmark_tool.py` |
| Native-vs-LangGraph workflow benchmark | `machina/research/workflow_benchmark.py` | Shipped research module (see the workflow-trial CI job) |
| Harness-evals judged LoCoMo pilot | `tools/harness_evals_pilot.py` | Operator tool, spends provider budget on purpose |
| Load-marked pytest modules (`pytest.mark.load`) | `tests/` (moved here) | Ordinary tests with a marker; deselect with `-m "not load"` |

Candidates for a future `benchmarks/` suite: retrieval recall/latency replays
with a pinned dataset, gateway restart/throughput measurements, and UI
interaction budgets. None of these are currently expressed as a runnable
benchmark harness (no `pytest-benchmark`, no `benchmarks/` in the source
repo), so nothing is bootstrapped here yet.
