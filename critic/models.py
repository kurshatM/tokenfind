"""Data models for the safety critic agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

_VALID_MATCH_MODES = frozenset({"boundary", "substring"})


class Verdict(Enum):
    """Final decision on whether output should be released."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True)
class SafetyRule:
    """A single safety rule that the critic checks against.

    This dataclass intentionally collocates rule definition (name, keywords,
    patterns) with match configuration (match_mode, frequency_scaling) for
    simplicity.  If the matching strategy needs to become more complex in
    the future, consider separating rule definition from evaluation logic.

    Attributes:
        name: Human-readable rule identifier (e.g. "violence").
        description: What this rule is meant to catch.
        keywords: Trigger words/phrases for this rule.
        patterns: Regex patterns to match structural content (e.g. SSNs).
            Patterns are compiled once at construction and cached internally.
        severity: How serious a violation is, from 0.0 (trivial) to 1.0 (critical).
        match_mode: ``"boundary"`` (default) uses word-boundary matching for
            single-word keywords and substring matching for multi-word phrases.
            ``"substring"`` uses naive substring matching for all keywords.
        frequency_scaling: When *True*, repeated hits within the same rule
            contribute additional severity via diminishing returns.
    """

    name: str
    description: str
    keywords: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    severity: float = 0.5
    match_mode: str = "boundary"
    frequency_scaling: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if not self.name:
            raise ValueError("name must not be empty")
        if self.match_mode not in _VALID_MATCH_MODES:
            raise ValueError(
                f"match_mode must be one of {_VALID_MATCH_MODES}, "
                f"got {self.match_mode!r}"
            )
        # Compile and cache regex patterns once.  Stored via
        # object.__setattr__ because the dataclass is frozen.  This field
        # is not declared, so it does not participate in __eq__/__hash__.
        compiled: list[re.Pattern[str]] = []
        for pat in self.patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern {pat!r} in rule {self.name!r}: {exc}"
                ) from exc
        object.__setattr__(self, "_compiled_patterns", tuple(compiled))

    @property
    def compiled_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Pre-compiled regex patterns (read-only, cached at construction)."""
        return self._compiled_patterns  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RuleMatch:
    """A single keyword or pattern hit against a rule.

    Attributes:
        rule: The rule that was triggered.
        matched_keyword: The keyword or pattern string that matched.
        matched_text: The actual text span that was matched.
        start: Start index in the original text.
        end: End index in the original text.
        match_type: ``"keyword"``, ``"pattern"``, ``"entropy"``,
            ``"protected_content"``, ``"protected_segment"``, or
            ``"prompt_risk"``.
    """

    rule: SafetyRule
    matched_keyword: str
    matched_text: str = ""
    start: int = 0
    end: int = 0
    match_type: str = "keyword"


@dataclass(frozen=True)
class ViolationDetail:
    """Structured per-rule breakdown for a review result.

    Attributes:
        rule_name: Name of the triggered rule.
        rule_description: Description of the triggered rule.
        severity: Base severity of the rule.
        severity_contributed: Actual severity contributed after frequency
            scaling (may exceed base if ``frequency_scaling`` is on).
        matches: All individual matches for this rule.
    """

    rule_name: str
    rule_description: str
    severity: float
    severity_contributed: float
    matches: tuple[RuleMatch, ...]


@dataclass
class ReviewResult:
    """The critic's full judgement on a piece of text.

    Attributes:
        verdict: ALLOW, FLAG, or BLOCK.
        score: Safety score from 0.0 (dangerous) to 1.0 (safe).
        violations: Flat list of every rule match that was triggered.
        explanation: Human-readable summary of the decision.
        violation_details: Structured per-rule breakdown.
        confidence: How confident the critic is in its verdict (0.0–1.0).
            For ALLOW verdicts with no violations, confidence defaults to
            1.0 (no ambiguity).  For non-ALLOW verdicts, confidence is
            computed from match density and reflects how much the downstream
            system should trust the FLAG/BLOCK decision.
        is_refusal: Whether the output was classified as an LLM refusal
            message.  Useful for downstream analysis and the paper.
        sanitized_text: Redacted copy of the input (only when sanitize=True).
    """

    verdict: Verdict
    score: float
    violations: list[RuleMatch] = field(default_factory=list)
    explanation: str = ""
    violation_details: list[ViolationDetail] = field(default_factory=list)
    confidence: float = 1.0
    is_refusal: bool = False
    sanitized_text: str | None = None
