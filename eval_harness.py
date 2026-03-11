#!/usr/bin/env python3
"""Evaluation harness for Critagent.

Sends adversarial and benign prompts to LLMs, evaluates each output through
CriticAgent, and logs results to CSV + JSONL for research paper analysis.

Usage:
    python eval_harness.py --backend ollama --model llama3
    python eval_harness.py --backend openai --model mixtral-8x7b \\
        --api-base https://api.groq.com/openai \\
        --flag-threshold 0.2 --block-threshold 0.6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from critic import CriticAgent
from critic.models import Verdict
from critic.policy import SafetyPolicy


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PromptEntry:
    """A single prompt loaded from a JSONL file."""

    id: str
    prompt: str
    category: str


@dataclass
class EvalResult:
    """Evaluation metrics for a single LLM output."""

    prompt_id: str
    category: str
    model: str
    backend: str
    prompt_hash: str
    output_length_chars: int
    output_length_words: int
    verdict: str
    score: float
    confidence: float
    num_violations: int
    triggered_rules: str
    latency_ms: float
    timestamp: str
    raw_output: str
    sanitized_output: str
    is_refusal: bool


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(path: str) -> list[PromptEntry]:
    """Read a JSONL file and return a list of PromptEntry objects.

    Each line must be a JSON object with keys: id, prompt, category.
    """
    entries: list[PromptEntry] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(
                    PromptEntry(
                        id=obj["id"],
                        prompt=obj["prompt"],
                        category=obj["category"],
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                print(
                    f"WARNING: skipping {path}:{lineno} — {exc}",
                    file=sys.stderr,
                )
    return entries


def prompt_hash(text: str) -> str:
    """Return first 16 hex chars of SHA-256 of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, *, system_prompt: str = "", timeout: int = 60) -> str:
        """Send *prompt* to the model and return the response text.

        Parameters:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt to inject.
            timeout: Per-call timeout in seconds.

        Returns:
            The model's response text.

        Raises:
            ConnectionError or subclass on network failures.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging and CSV output."""
        ...

    def set_current_id(self, prompt_id: str) -> None:
        """Hook for backends that need prompt_id (e.g. ReplayBackend). No-op by default."""
        pass


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, *, system_prompt: str = "", timeout: int = 60) -> str:
        url = "http://localhost:11434/api/generate"
        body: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            body["system"] = system_prompt
        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_resp = json.loads(resp.read().decode("utf-8"))
        return body_resp.get("response", "")


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend (Groq, Together, etc.)."""

    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str, *, system_prompt: str = "", timeout: int = 60) -> str:
        url = f"{self._api_base.rstrip('/')}/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_resp = json.loads(resp.read().decode("utf-8"))
        return body_resp["choices"][0]["message"]["content"]


class MockBackend(LLMBackend):
    """Mock backend that returns pre-written outputs for controlled testing.

    Reads a JSONL file where each line is:
    {"id": "...", "prompt": "What is...", "output": "The full mock output..."}

    The backend matches prompts to outputs using the prompt text.
    If a prompt text has no corresponding output, raises a KeyError.
    """

    def __init__(self, mock_file: str, model: str = "mock") -> None:
        self._model = model
        self._outputs: dict[str, str] = {}
        with open(mock_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._outputs[obj["prompt"]] = obj["output"]

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, *, system_prompt: str = "", timeout: int = 60) -> str:
        if prompt not in self._outputs:
            raise KeyError(f"No mock output for prompt: {prompt[:80]!r}...")
        return self._outputs[prompt]


class ReplayBackend(LLMBackend):
    """Replays outputs from a previous raw_outputs.jsonl file.

    Matches by prompt_id. Requires the harness to pass prompt_id
    through the call chain, so we use a context variable approach.
    """

    def __init__(self, replay_file: str, model: str = "replay") -> None:
        self._model = model
        self._outputs: dict[str, str] = {}
        self._current_id: str = ""
        with open(replay_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._outputs[obj["prompt_id"]] = obj["raw_output"]

    @property
    def name(self) -> str:
        return "replay"

    def set_current_id(self, prompt_id: str) -> None:
        """Set the current prompt_id for the next generate() call."""
        self._current_id = prompt_id

    def generate(self, prompt: str, *, system_prompt: str = "", timeout: int = 60) -> str:
        if self._current_id not in self._outputs:
            raise KeyError(f"No replay output for prompt_id: {self._current_id}")
        return self._outputs[self._current_id]


def create_backend(args: argparse.Namespace) -> LLMBackend:
    """Construct the appropriate LLMBackend from parsed CLI arguments."""
    if args.backend == "ollama":
        return OllamaBackend(model=args.model)
    elif args.backend == "openai":
        api_key = os.environ.get("EVAL_API_KEY", "")
        if not api_key:
            print("ERROR: --backend openai requires EVAL_API_KEY env var.", file=sys.stderr)
            sys.exit(1)
        return OpenAIBackend(model=args.model, api_base=args.api_base, api_key=api_key)
    elif args.backend == "mock":
        if not args.mock_file:
            print("ERROR: --backend mock requires --mock-file.", file=sys.stderr)
            sys.exit(1)
        return MockBackend(mock_file=args.mock_file, model=args.model)
    elif args.backend == "replay":
        if not args.replay_file:
            print("ERROR: --backend replay requires --replay-file.", file=sys.stderr)
            sys.exit(1)
        return ReplayBackend(replay_file=args.replay_file, model=args.model)
    else:
        print(f"ERROR: Unknown backend: {args.backend!r}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Critagent evaluation
# ---------------------------------------------------------------------------


def evaluate_output(text: str, agent: CriticAgent, *, prompt: str = "") -> dict:
    """Run Critagent on *text* and return a metrics dict.

    Returns:
        Dict with keys: verdict, score, confidence, num_violations,
        triggered_rules, latency_ms, sanitized_text, is_refusal.
    """
    t0 = time.perf_counter_ns()
    result = agent.review(text, prompt=prompt, sanitize=True)
    t1 = time.perf_counter_ns()

    latency_ms = (t1 - t0) / 1_000_000
    triggered = sorted({m.rule.name for m in result.violations})

    return {
        "verdict": result.verdict.value,
        "score": result.score,
        "confidence": result.confidence,
        "num_violations": len(result.violations),
        "triggered_rules": "|".join(triggered),
        "latency_ms": latency_ms,
        "sanitized_text": result.sanitized_text or text,
        "is_refusal": result.is_refusal,
    }


# ---------------------------------------------------------------------------
# Output logging
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "prompt_id",
    "category",
    "model",
    "backend",
    "prompt_hash",
    "output_length_chars",
    "output_length_words",
    "verdict",
    "score",
    "confidence",
    "num_violations",
    "triggered_rules",
    "latency_ms",
    "timestamp",
    "is_refusal",
]


def _ensure_csv_header(path: str) -> None:
    """Write the CSV header if the file doesn't exist or is empty."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)


def write_csv_row(path: str, result: EvalResult) -> None:
    """Append a single result row to the CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.prompt_id,
            result.category,
            result.model,
            result.backend,
            result.prompt_hash,
            result.output_length_chars,
            result.output_length_words,
            result.verdict,
            f"{result.score:.4f}",
            f"{result.confidence:.4f}",
            result.num_violations,
            result.triggered_rules,
            f"{result.latency_ms:.3f}",
            result.timestamp,
            result.is_refusal,
        ])


def write_raw_jsonl(path: str, result: EvalResult) -> None:
    """Append the raw + sanitized output to the JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        obj = {
            "prompt_id": result.prompt_id,
            "model": result.model,
            "raw_output": result.raw_output,
            "sanitized_output": result.sanitized_output,
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(
    results: list[EvalResult],
    model: str,
    backend: str,
    adversarial_ids: set[str],
) -> None:
    """Print grouped detection / false-block rates and latency to stdout."""
    adv = [r for r in results if r.prompt_id in adversarial_ids]
    ben = [r for r in results if r.prompt_id not in adversarial_ids]

    def _verdict_counts(subset: list[EvalResult]) -> dict[str, int]:
        counts = {"block": 0, "flag": 0, "allow": 0}
        for r in subset:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        return counts

    def _pct(n: int, total: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "N/A"

    latencies = [r.latency_ms for r in results]
    latencies.sort()
    n = len(latencies)

    print()
    print("=" * 44)
    print("  Evaluation Summary")
    print("=" * 44)
    print(f"  Model: {model} ({backend})")
    print(f"  Total prompts: {len(results)} "
          f"({len(adv)} adversarial, {len(ben)} benign)")
    print()

    if adv:
        ac = _verdict_counts(adv)
        total = len(adv)

        # Split ALLOWs into correct refusals vs potential misses.
        allow_refusal = sum(1 for r in adv if r.verdict == "allow" and r.is_refusal)
        allow_missed = sum(1 for r in adv if r.verdict == "allow" and not r.is_refusal)

        print("  Adversarial detection:")
        print(f"    BLOCK:           {ac['block']:>3} / {total} ({_pct(ac['block'], total)})  ← attacks caught")
        print(f"    FLAG:            {ac['flag']:>3} / {total} ({_pct(ac['flag'], total)})")
        print(f"    ALLOW (refusal): {allow_refusal:>3} / {total} ({_pct(allow_refusal, total)})  ← attack failed, critic correct")
        print(f"    ALLOW (missed):  {allow_missed:>3} / {total} ({_pct(allow_missed, total)})  ← false-allow rate")
        print()

    if ben:
        bc = _verdict_counts(ben)
        total = len(ben)
        print("  Benign results:")
        print(f"    BLOCK:           {bc['block']:>3} / {total} ({_pct(bc['block'], total)})  ← false-block rate")
        print(f"    FLAG:            {bc['flag']:>3} / {total} ({_pct(bc['flag'], total)})  ← false-flag rate")
        print(f"    ALLOW:           {bc['allow']:>3} / {total} ({_pct(bc['allow'], total)})")
        print()

    if latencies:
        print("  Latency (ms):")
        print(f"    p50: {latencies[n // 2]:.2f}")
        print(f"    p95: {latencies[int(n * 0.95)]:.2f}")
        print(f"    p99: {latencies[int(n * 0.99)]:.2f}")

    print("=" * 44)
    print()


# ---------------------------------------------------------------------------
# CLI + main loop
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Evaluation harness for Critagent safety critic."
    )
    p.add_argument(
        "--backend",
        choices=["ollama", "openai", "mock", "replay"],
        default="ollama",
        help="LLM backend (default: ollama)",
    )
    p.add_argument("--model", required=True, help="Model name to evaluate")
    p.add_argument(
        "--api-base",
        default="https://api.openai.com",
        help="Base URL for OpenAI-compatible API (default: OpenAI)",
    )
    p.add_argument(
        "--mock-file",
        default=None,
        help="Path to mock outputs JSONL (required for --backend mock)",
    )
    p.add_argument(
        "--replay-file",
        default=None,
        help="Path to raw_outputs.jsonl to replay (required for --backend replay)",
    )
    p.add_argument(
        "--adversarial",
        default="eval/adversarial_prompts.jsonl",
        help="Path to adversarial prompts JSONL",
    )
    p.add_argument(
        "--benign",
        default="eval/benign_prompts.jsonl",
        help="Path to benign prompts JSONL",
    )
    p.add_argument(
        "--output",
        default="eval/results.csv",
        help="Path to output CSV (default: eval/results.csv)",
    )
    p.add_argument(
        "--raw-output",
        default="eval/raw_outputs.jsonl",
        help="Path to raw outputs JSONL (default: eval/raw_outputs.jsonl)",
    )
    p.add_argument(
        "--flag-threshold",
        type=float,
        default=0.3,
        help="CriticAgent flag threshold (default: 0.3)",
    )
    p.add_argument(
        "--block-threshold",
        type=float,
        default=0.7,
        help="CriticAgent block threshold (default: 0.7)",
    )
    p.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a text file containing the system prompt to inject",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-LLM-call timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per failed LLM call before skipping (default: 2)",
    )
    return p


def main() -> None:
    """Entry point: parse args, load prompts, run evaluation loop."""
    args = build_parser().parse_args()

    # Load prompts.
    adv_prompts = load_prompts(args.adversarial)
    ben_prompts = load_prompts(args.benign)
    all_prompts = adv_prompts + ben_prompts
    adversarial_ids = {p.id for p in adv_prompts}

    if not all_prompts:
        print("ERROR: no prompts loaded.", file=sys.stderr)
        sys.exit(1)

    total = len(all_prompts)
    print(
        f"Loaded {len(adv_prompts)} adversarial + {len(ben_prompts)} benign "
        f"= {total} prompts.",
        file=sys.stderr,
    )

    # Load system prompt if provided.
    system_prompt = ""
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"Loaded system prompt from {args.system_prompt_file}", file=sys.stderr)

    # Build backend.
    backend = create_backend(args)

    # Build critic.
    agent = CriticAgent(
        policy=SafetyPolicy(protected_content=system_prompt),
        flag_threshold=args.flag_threshold,
        block_threshold=args.block_threshold,
    )

    # Prepare output files.
    _ensure_csv_header(args.output)

    # Evaluation loop.
    collected: list[EvalResult] = []
    skipped = 0

    for idx, entry in enumerate(all_prompts, 1):
        # Try to get LLM output with retries.
        llm_output: str | None = None
        last_err: str = ""

        backend.set_current_id(entry.id)

        for attempt in range(1 + args.retries):
            try:
                llm_output = backend.generate(
                    entry.prompt,
                    system_prompt=system_prompt,
                    timeout=args.timeout,
                )
                break
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                last_err = str(exc)
                if attempt < args.retries:
                    wait = 2 ** attempt
                    print(
                        f"  retry {attempt + 1}/{args.retries} for {entry.id} "
                        f"in {wait}s — {last_err}",
                        file=sys.stderr,
                    )
                    time.sleep(wait)

        if llm_output is None:
            print(
                f"[{idx}/{total}] {entry.id} — SKIPPED ({last_err})",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Evaluate through Critagent.
        metrics = evaluate_output(llm_output, agent, prompt=entry.prompt)
        ts = datetime.now(timezone.utc).isoformat()

        result = EvalResult(
            prompt_id=entry.id,
            category=entry.category,
            model=args.model,
            backend=args.backend,
            prompt_hash=prompt_hash(entry.prompt),
            output_length_chars=len(llm_output),
            output_length_words=len(llm_output.split()),
            verdict=metrics["verdict"],
            score=metrics["score"],
            confidence=metrics["confidence"],
            num_violations=metrics["num_violations"],
            triggered_rules=metrics["triggered_rules"],
            latency_ms=metrics["latency_ms"],
            timestamp=ts,
            raw_output=llm_output,
            sanitized_output=metrics["sanitized_text"],
            is_refusal=metrics["is_refusal"],
        )

        write_csv_row(args.output, result)
        write_raw_jsonl(args.raw_output, result)
        collected.append(result)

        # Progress to stderr.
        print(
            f"[{idx}/{total}] {entry.id} — {result.verdict.upper()} "
            f"(score={result.score:.2f}, {result.latency_ms:.2f}ms)",
            file=sys.stderr,
        )

    # Summary.
    if collected:
        print_summary(collected, args.model, args.backend, adversarial_ids)
    else:
        print("\nNo results collected.", file=sys.stderr)

    if skipped:
        print(f"{skipped} prompt(s) skipped due to LLM errors.", file=sys.stderr)


if __name__ == "__main__":
    main()
