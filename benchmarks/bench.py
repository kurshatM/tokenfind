#!/usr/bin/env python3
"""Benchmarking harness for CriticAgent.

Generates synthetic text samples (mix of clean + violation-containing),
runs CriticAgent.review() on each, and reports latency statistics.

Usage:
    python benchmarks/bench.py              # default 1000 samples
    python benchmarks/bench.py --samples 5000
    python benchmarks/bench.py --samples 2000 --sanitize
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time

# Ensure the project root is importable.
sys.path.insert(0, ".")

from critic import CriticAgent, SafetyPolicy


# ---------------------------------------------------------------------------
# Synthetic sample generation
# ---------------------------------------------------------------------------

_CLEAN_TEMPLATES = (
    "The weather today is sunny with a high of {n} degrees.",
    "Our quarterly report shows a {n}% increase in revenue.",
    "The package has been shipped and will arrive in {n} days.",
    "Please refer to section {n} of the documentation for details.",
    "The team completed {n} story points during the last sprint.",
    "Version {n}.0 includes significant performance improvements.",
    "The database migration completed successfully in {n} seconds.",
    "Customer satisfaction scores improved by {n} points this quarter.",
)

_VIOLATION_TEMPLATES = (
    "He threatened to stab someone outside the building.",
    "My SSN is 123-45-6789, please don't share it.",
    "Instructions on how to end your life permanently.",
    "The racist bigot kept shouting hate speech at the crowd.",
    "Here is credit card 4111-1111-1111-1111 for payment.",
    "Please hack into the company server to get the data.",
    "She said she wanted to kill you after the argument.",
    "The bomb threat was called in at noon, causing panic.",
    "Your API key is sk_live_abcdefghij1234567890xx for this account.",
    "He tried to smuggle counterfeit goods across the border.",
)


def _generate_samples(n: int, violation_ratio: float = 0.3) -> list[str]:
    """Return *n* synthetic text samples.

    Approximately *violation_ratio* of them contain violations; the rest
    are clean.  Clean samples are padded with filler to vary length.
    """
    rng = random.Random(42)  # deterministic for reproducibility
    samples: list[str] = []
    for _ in range(n):
        if rng.random() < violation_ratio:
            samples.append(rng.choice(_VIOLATION_TEMPLATES))
        else:
            # Build a multi-sentence clean sample (10–100 words).
            sentences = rng.randint(1, 5)
            parts = [
                rng.choice(_CLEAN_TEMPLATES).format(n=rng.randint(1, 999))
                for _ in range(sentences)
            ]
            samples.append(" ".join(parts))
    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    samples: list[str],
    *,
    sanitize: bool = False,
) -> dict[str, float]:
    """Run the critic over *samples* and return latency stats in µs."""
    policy = SafetyPolicy()
    agent = CriticAgent(policy=policy)

    # Warm-up pass (compile regex, JIT, etc.).
    for s in samples[:50]:
        agent.review(s, sanitize=sanitize)

    latencies_ns: list[int] = []
    for s in samples:
        t0 = time.perf_counter_ns()
        agent.review(s, sanitize=sanitize)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    latencies_us = [ns / 1_000 for ns in latencies_ns]
    latencies_us.sort()

    n = len(latencies_us)
    return {
        "samples": n,
        "p50_us": latencies_us[n // 2],
        "p95_us": latencies_us[int(n * 0.95)],
        "p99_us": latencies_us[int(n * 0.99)],
        "mean_us": statistics.mean(latencies_us),
        "stdev_us": statistics.stdev(latencies_us) if n > 1 else 0.0,
        "total_ms": sum(latencies_us) / 1_000,
        "throughput_per_sec": n / (sum(latencies_ns) / 1e9),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CriticAgent latency benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--sanitize", action="store_true", help="Enable sanitization")
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic samples …")
    samples = _generate_samples(args.samples)

    print(f"Running benchmark (sanitize={args.sanitize}) …\n")
    stats = run_benchmark(samples, sanitize=args.sanitize)

    print("=" * 52)
    print("  CriticAgent Latency Benchmark")
    print("=" * 52)
    print(f"  Samples:           {stats['samples']:>10,}")
    print(f"  Sanitize:          {str(args.sanitize):>10}")
    print(f"  ─────────────────────────────────")
    print(f"  p50  latency:      {stats['p50_us']:>10.1f} µs")
    print(f"  p95  latency:      {stats['p95_us']:>10.1f} µs")
    print(f"  p99  latency:      {stats['p99_us']:>10.1f} µs")
    print(f"  Mean latency:      {stats['mean_us']:>10.1f} µs")
    print(f"  Stdev:             {stats['stdev_us']:>10.1f} µs")
    print(f"  ─────────────────────────────────")
    print(f"  Total time:        {stats['total_ms']:>10.1f} ms")
    print(f"  Throughput:        {stats['throughput_per_sec']:>10,.0f} reviews/sec")
    print("=" * 52)


if __name__ == "__main__":
    main()
