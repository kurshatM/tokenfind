"""CriticAgent — the main entry point for safety reviews."""

from __future__ import annotations

import math
from collections import defaultdict

from critic.models import ReviewResult, RuleMatch, SafetyRule, Verdict, ViolationDetail
from critic.policy import SafetyPolicy, _is_refusal


class CriticAgent:
    """A lightweight critic that reviews text and decides ALLOW / FLAG / BLOCK.

    Uses purely deterministic, rule-based analysis — no LLM, no ML models.

    Parameters:
        policy: A :class:`SafetyPolicy` to evaluate against.
                Defaults to the built-in policy when *None*.
        flag_threshold: Cumulative severity at or above which the verdict
                        becomes FLAG (default 0.3).
        block_threshold: Cumulative severity at or above which the verdict
                         becomes BLOCK (default 0.7).
    """

    def __init__(
        self,
        policy: SafetyPolicy | None = None,
        *,
        flag_threshold: float = 0.3,
        block_threshold: float = 0.7,
    ) -> None:
        if flag_threshold < 0 or block_threshold < 0:
            raise ValueError("Thresholds must be non-negative.")
        if flag_threshold > block_threshold:
            raise ValueError(
                f"flag_threshold ({flag_threshold}) must be "
                f"<= block_threshold ({block_threshold})."
            )
        self._policy = policy or SafetyPolicy()
        self._flag_threshold = flag_threshold
        self._block_threshold = block_threshold

    # -- public API ---------------------------------------------------------

    def review(self, text: str, *, prompt: str = "", sanitize: bool = False) -> ReviewResult:
        """Review *text* and return a :class:`ReviewResult`.

        Parameters:
            text: The complete output text to evaluate.
            prompt: The user prompt that produced the output (optional).
                Used for prompt-aware scoring: if the prompt itself
                triggers safety rules and the output is not a refusal,
                a severity boost is applied.
            sanitize: When *True*, populate ``sanitized_text`` with a redacted
                copy of *text* (matched spans replaced with ``[REDACTED]``).

        The safety **score** is ``1 - cumulative_severity`` clamped to [0, 1].
        The **verdict** is decided by comparing cumulative severity to the
        configured thresholds.
        """
        if not text or not text.strip():
            return ReviewResult(
                verdict=Verdict.ALLOW,
                score=1.0,
                violations=[],
                explanation="Empty input — nothing to review.",
            )

        refusal = _is_refusal(text)
        matches: list[RuleMatch] = self._policy.evaluate(
            text, is_refusal=refusal
        )

        # --- Prompt-aware scoring ---
        # If the prompt itself triggers safety rules and the output is NOT
        # a refusal, apply a modest severity boost.  This catches cases where
        # the prompt is clearly adversarial but the output rephrases its
        # compliance to avoid the same keywords.
        prompt_match: RuleMatch | None = None
        if prompt and not refusal:
            prompt_matches = self._policy.evaluate_prompt(prompt)
            if prompt_matches:
                prompt_severity = sum(m.rule.severity for m in prompt_matches)
                boost = min(0.3, prompt_severity * 0.3)
                if boost > 0:
                    prompt_rule = SafetyRule(
                        name="prompt_risk_boost",
                        description="User prompt contains safety-relevant "
                                    "keywords; output did not refuse.",
                        severity=boost,
                    )
                    prompt_match = RuleMatch(
                        rule=prompt_rule,
                        matched_keyword=f"prompt_boost:{boost:.2f}",
                        matched_text="",
                        start=0,
                        end=0,
                        match_type="prompt_risk",
                    )
                    matches.append(prompt_match)

        if not matches:
            return ReviewResult(
                verdict=Verdict.ALLOW,
                score=1.0,
                violations=[],
                explanation="No safety violations detected.",
                is_refusal=refusal,
                sanitized_text=text if sanitize else None,
            )

        details = self._build_violation_details(matches)
        cumulative = sum(d.severity_contributed for d in details)
        score = max(0.0, min(1.0, 1.0 - cumulative))
        verdict = self._decide_verdict(cumulative)
        explanation = self._build_explanation(matches, verdict)
        confidence = self._compute_confidence(text, matches)
        sanitized = self._sanitize(text, matches) if sanitize else None

        return ReviewResult(
            verdict=verdict,
            score=score,
            violations=matches,
            explanation=explanation,
            violation_details=details,
            confidence=confidence,
            is_refusal=refusal,
            sanitized_text=sanitized,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_violation_details(
        matches: list[RuleMatch],
    ) -> list[ViolationDetail]:
        """Group matches by rule and compute per-rule severity contribution."""
        by_rule: dict[str, list[RuleMatch]] = defaultdict(list)
        first_match_by_rule: dict[str, RuleMatch] = {}
        for m in matches:
            by_rule[m.rule.name].append(m)
            first_match_by_rule.setdefault(m.rule.name, m)

        details: list[ViolationDetail] = []
        for rule_name, rule_matches in by_rule.items():
            rule = first_match_by_rule[rule_name].rule
            count = len(rule_matches)

            if rule.frequency_scaling and count > 1:
                # Diminishing-returns: base + 0.1 * ln(count), capped at 2×.
                contributed = min(
                    rule.severity * 2.0,
                    rule.severity + 0.1 * math.log(count),
                )
            else:
                contributed = rule.severity

            details.append(
                ViolationDetail(
                    rule_name=rule.name,
                    rule_description=rule.description,
                    severity=rule.severity,
                    severity_contributed=contributed,
                    matches=tuple(rule_matches),
                )
            )
        return details


    def _decide_verdict(self, cumulative: float) -> Verdict:
        if cumulative >= self._block_threshold:
            return Verdict.BLOCK
        if cumulative >= self._flag_threshold:
            return Verdict.FLAG
        return Verdict.ALLOW

    @staticmethod
    def _build_explanation(matches: list[RuleMatch], verdict: Verdict) -> str:
        rule_names = sorted({m.rule.name for m in matches})
        rules_str = ", ".join(rule_names)
        return (
            f"Verdict: {verdict.value.upper()}. "
            f"Triggered rules: {rules_str}."
        )

    @staticmethod
    def _compute_confidence(text: str, matches: list[RuleMatch]) -> float:
        """Compute confidence in the verdict based on match density.

        Uses a logarithmic curve ``1 - 1 / (1 + ln(1 + density * 10))``
        for a smooth falloff: high density saturates near 1.0, low density
        drops gradually rather than falling off a cliff.

        Returns a float in [0.1, 1.0].
        """
        word_count = max(1, len(text.split()))
        density = len(matches) / word_count
        raw = 1.0 - 1.0 / (1.0 + math.log(1.0 + density * 10))
        return max(0.1, min(1.0, raw))

    @staticmethod
    def _sanitize(text: str, matches: list[RuleMatch]) -> str:
        """Return a copy of *text* with all matched spans replaced by
        ``[REDACTED]``.

        Spans are processed right-to-left so earlier indices stay valid.
        Overlapping spans are merged before replacement.
        """
        if not matches:
            return text

        # Collect and sort spans, then merge overlaps.
        spans = sorted(
            {(m.start, m.end) for m in matches}, key=lambda s: s[0]
        )
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Replace right-to-left.
        result = text
        for start, end in reversed(merged):
            result = result[:start] + "[REDACTED]" + result[end:]
        return result
