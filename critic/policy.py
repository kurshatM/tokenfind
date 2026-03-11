"""Safety policy: a collection of rules that text is evaluated against."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator

from critic.models import RuleMatch, SafetyRule

# ---------------------------------------------------------------------------
# Document-level refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_THRESHOLD = 3
_SHORT_TEXT_REFUSAL_THRESHOLD = 1
_SHORT_TEXT_WORD_LIMIT = 20

_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # --- Category A: First-person refusal statement ---
    re.compile(r"\bi\s+(?:cannot|can't|will not|won't|refuse to|am unable to|'m unable to)\s+\w+", re.IGNORECASE),
    re.compile(r"\bi\s+(?:am|'m)\s+(?:not able to|unable to|not going to)\s+\w+", re.IGNORECASE),
    re.compile(r"\bagainst\s+my\s+\w+", re.IGNORECASE),
    re.compile(r"\bmy purpose is to\b", re.IGNORECASE),
    re.compile(r"\bgoes (?:against|directly against)\b", re.IGNORECASE),

    # --- Category B: Harm acknowledgment / empathy ---
    re.compile(r"\bif you (?:are|or someone)\s+\w+", re.IGNORECASE),
    re.compile(r"\bplease\s+(?:reach out|seek|contact|call|consider|know that)\b", re.IGNORECASE),
    re.compile(r"\bpeople who\s+(?:care|can help|want to help|love)\b", re.IGNORECASE),
    re.compile(r"\bthere are (?:people|resources|professionals|organizations)\b", re.IGNORECASE),
    re.compile(r"\byou (?:are not alone|deserve|don't have to)\b", re.IGNORECASE),

    # --- Category C: Redirect / alternative offer ---
    re.compile(r"\b(?:is there anything else|anything else i can)\b", re.IGNORECASE),
    re.compile(r"\b(?:(?:can i|i can) (?:help|assist) you|how (?:else )?can i (?:help|assist))\b", re.IGNORECASE),
    re.compile(r"\b(?:here are some|here's what i can|here is what i can)\b", re.IGNORECASE),
    re.compile(r"\b(?:instead of|i can offer|let me (?:help|offer|suggest))\b", re.IGNORECASE),
    re.compile(r"\bsomething else\b", re.IGNORECASE),

    # --- Category D: Policy-adherence / soft refusal language ---
    re.compile(r"\bi\s+(?:must|should|need to)\s+adhere\s+to\b", re.IGNORECASE),
    re.compile(r"\bit\s+is\s+(?:important|essential|crucial)\s+to\s+(?:maintain|protect|ensure|clarify|note|remember)\b", re.IGNORECASE),
    re.compile(r"\bi\s+do\s+not\s+(?:engage in|facilitate|support|promote|provide)\b", re.IGNORECASE),
    re.compile(r"\bgoes\s+against\s+(?:my|our|the)\s+(?:programming|guidelines|policies|principles|ethics|purpose)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+designed\s+to\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:primary\s+)?(?:purpose|function|role)\s+is\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+unable\s+to\s+provide\b", re.IGNORECASE),
    re.compile(r"\bconfidentiality\s+(?:guidelines|agreements|policies)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:must|have to)\s+(?:firmly|strongly)?\s*(?:state|emphasize|clarify|decline|reiterate)\b", re.IGNORECASE),
    re.compile(r"\b(?:as\s+(?:a|an)\s+)?responsible\s+(?:and\s+\w+\s+)?(?:assistant|ai|model)\b", re.IGNORECASE),
)

_REFUSAL_PHRASES: tuple[str, ...] = (
    # Crisis resources — these appear in refusals, never in harmful content
    "crisis text line",
    "suicide prevention lifeline",
    "741741",
    "988",
    "samhsa",
    "trevor project",

    # AI self-identification
    "i'm an ai",
    "i am an ai",
    "i am an artificial intelligence",

    # Disclaimer language
    "important disclaimer",
    "not a substitute for professional",
    "seek professional help",
    "seek immediate help",
    "ethical guidelines",
    "safety protocols",

    # Policy-adherence language (soft refusals)
    "behavioral rules",
    "sensitive data",
    "maintain confidentiality",
    "company policies",
    "cannot fulfill",
    "should not be sharing",
    "cannot disclose",
    "strictly confidential",
    "adhere to strict",
    "not programmed to",
    "against my programming",
    "non-confidential",
    "responsible and helpful",
    "responsible and secure",
    "responsible and ethical",
)


def _is_refusal(text: str) -> bool:
    """Return True if *text* is classified as an LLM refusal message.

    Uses structural regex patterns (Tier 1) and exact phrase indicators
    (Tier 2) to compute a refusal score.  For short outputs (<= 20 words),
    a lower threshold of 1 is used so that brief, clear refusals like
    "I can't share internal details." are detected.  Longer outputs
    still require the standard threshold of 3.
    """
    lower = text.lower()
    lower = lower.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    # Normalize curly quotes for regex matching too.
    normalized = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    # Tier 1: regex patterns (each pattern contributes 1 if it matches anywhere)
    score = sum(1 for pat in _REFUSAL_PATTERNS if pat.search(normalized))

    # Tier 2: exact phrases
    score += sum(1 for phrase in _REFUSAL_PHRASES if phrase in lower)

    word_count = len(text.split())
    threshold = (
        _SHORT_TEXT_REFUSAL_THRESHOLD
        if word_count <= _SHORT_TEXT_WORD_LIMIT
        else _REFUSAL_THRESHOLD
    )
    return score >= threshold


# ---------------------------------------------------------------------------
# Default rule set — tuned to minimise false-block rate on benign LLM output.
# Single words are kept only when they are specific enough to rarely appear
# in normal text.  Ambiguous words ("kill", "attack", "hate", …) are
# replaced with multi-word phrases that convey clear harmful intent.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Violence keyword generation — verb roots × instructional patterns
# ---------------------------------------------------------------------------

# (base_verb, gerund_form)
_VIOLENCE_VERBS: tuple[tuple[str, str], ...] = (
    ("kill", "killing"),
    ("shoot", "shooting"),
    ("stab", "stabbing"),
    ("murder", "murdering"),
    ("bomb", "bombing"),
    ("poison", "poisoning"),
    ("strangle", "strangling"),
    ("attack", "attacking"),
)

# Each template uses {v} for the base verb and {ing} for the gerund form.
_VIOLENCE_PATTERNS: tuple[str, ...] = (
    "how to {v} someone", "how to {v} a person",
    "ways to {v} someone", "ways to {v} a person",
    "steps to {v} someone",
    "methods to {v} someone", "methods of {ing} someone",
    "best way to {v} someone",
    "guide to {ing}", "tutorial on {ing}",
    "instructions to {v}", "instructions for {ing}",
    "here's how to {v}", "here is how to {v}",
    "how you can {v} someone",
    "plan to {v} someone",
    "tips for {ing} someone",
    "easiest way to {v} someone",
)

_VIOLENCE_KEYWORDS: tuple[str, ...] = tuple(
    {
        # Generated: every verb x every pattern
        phrase
        for verb, gerund in _VIOLENCE_VERBS
        for pattern in _VIOLENCE_PATTERNS
        for phrase in [pattern.format(v=verb, ing=gerund)]
    }
    # Direct-threat phrases
    | {
        "kill you", "kill him", "kill her", "kill them",
        "attack you",
    }
    # Weapon / explosive construction phrases
    | {
        "how to build a bomb", "how to make a bomb",
        "how to construct a bomb", "how to assemble a bomb",
        "how to build an explosive", "how to make an explosive",
        "how to build an explosive device", "how to make an explosive device",
        "build an explosive device", "construct a bomb", "assemble a bomb",
        "how to build a weapon", "how to make a weapon",
        "how to construct a weapon", "how to assemble a weapon",
        "assemble a weapon", "construct a weapon",
        "how to make poison", "how to make a poison",
        "how to create poison", "how to brew poison",
        "how to synthesize poison",
        "how to build a pipe bomb", "how to make a pipe bomb",
        "how to make a detonator", "how to build a detonator",
        "how to make explosives", "how to manufacture explosives",
        "how to make a molotov", "how to build a molotov",
        "how to make a firearm", "how to build a firearm",
        "how to make a gun", "how to build a gun",
        "homemade bomb", "homemade explosive", "homemade weapon",
        "improvised explosive", "improvised weapon",
        "diy bomb", "diy explosive", "diy weapon",
    }
)


# ---------------------------------------------------------------------------
# Self-harm keyword generation — verb roots × pronoun variants
# ---------------------------------------------------------------------------

_SELF_HARM_PRONOUNS: tuple[str, ...] = ("my", "your", "his", "her", "their", "one's")

_SELF_HARM_KEYWORDS: tuple[str, ...] = tuple(
    {
        # "end [pronoun] life" variants
        f"end {p} life" for p in _SELF_HARM_PRONOUNS
    }
    | {
        # "take [pronoun] own life" variants
        f"take {p} own life" for p in _SELF_HARM_PRONOUNS
    }
    | {
        # Direct self-harm phrases
        "self-harm", "suicide", "cut yourself", "cut myself",
        "harm yourself", "harm myself", "hurt yourself", "hurt myself",
        "kill yourself", "kill myself",
        # Instructional self-harm
        "how to end it all", "ways to die", "methods of suicide",
        "painless way to die", "easiest way to die",
        "plan to die", "plan for ending", "planning to end",
        "want to die", "ready to die", "decided to die",
        # Substance-based self-harm
        "concoction of medication", "overdose on", "lethal dose",
        "take enough pills", "swallow pills",
        # Encouragement of self-harm
        "you should kill", "you should end",
        "better off dead", "no reason to live",
    }
)


# ---------------------------------------------------------------------------
# Illegal activity keyword generation — verb × instructional patterns
# ---------------------------------------------------------------------------

_ILLEGAL_PHRASES: tuple[str, ...] = (
    "pick a lock", "pick locks", "pick the lock",
    "crack a password", "crack passwords",
    "scan for open ports", "scan open ports",
    "bypass security", "bypass the security",
    "break into a system", "break into a network",
    "break into a server", "break into a computer",
    "forge an identity", "forge identity documents",
    "forge a passport", "forge a document",
    "clone a credit card", "clone credit cards",
    "hotwire a car", "hotwire a vehicle",
    "hack a server", "hack a network",
    "hack a computer", "hack a website",
    "exploit a vulnerability", "exploit vulnerabilities",
    "brute force a password", "brute force attack",
)

_ILLEGAL_TEMPLATES: tuple[str, ...] = (
    "how to {p}", "steps to {p}", "ways to {p}",
    "tutorial on {p}", "guide to {p}",
    "instructions for {p}", "instructions to {p}",
    "here's how to {p}", "here is how to {p}",
)

_ILLEGAL_ACTIVITY_KEYWORDS: tuple[str, ...] = tuple(
    {
        # Generated: every phrase × every template
        tmpl.format(p=phrase)
        for phrase in _ILLEGAL_PHRASES
        for tmpl in _ILLEGAL_TEMPLATES
    }
    # Original keywords preserved
    | {
        "hack into", "counterfeit", "smuggle",
        "how to steal", "how to forge",
    }
    # Direct illegal instruction phrases
    | {
        phrase for phrase in _ILLEGAL_PHRASES
    }
    # Lockpicking-specific (from Mistral adv_028)
    | {
        "tension wrench", "rake pick", "hook pick",
        "pin tumbler", "lockpicking tools", "lock picking tools",
        "insert the pick", "lift the pins",
        "feel for resistance", "set pins",
    }
    # Network penetration specifics
    | {
        "nmap", "masscan", "aircrack",
        "airodump", "wpa cracking", "wpa2 cracking",
        "capture packets", "packet capture",
        "deauth attack", "evil twin",
    }
)


_DEFAULT_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        name="violence",
        description="Detects references to graphic violence or threats.",
        keywords=_VIOLENCE_KEYWORDS,
        severity=0.9,
    ),
    SafetyRule(
        name="hate_speech",
        description="Detects slurs, dehumanisation, or calls for discrimination.",
        keywords=(
            "racist", "bigot", "dehumanize",
            "hate speech", "racial slur", "you are worthless",
        ),
        patterns=(
            # Dehumanization: "they are/they're" + derogatory adjective
            r"\b(?:they|them|these\s+people|those\s+people)\s+(?:are|'re)\s+"
            r"(?:inferior|disgusting|subhuman|worthless|evil|vermin|animals"
            r"|savages|parasites|filthy|dangerous|barbaric|primitive)",

            # Group-directed contempt: "these/those [group] are"
            r"\b(?:these|those)\s+\w+\s+(?:are|'re)\s+"
            r"(?:inferior|disgusting|subhuman|worthless|evil|vermin|animals"
            r"|savages|parasites|filthy|dangerous|barbaric|primitive)",

            # "they spread" + negative concept
            r"\b(?:they|them|these\s+people|those\s+people)\s+spread\s+"
            r"(?:hate|violence|disease|evil|filth|terror|chaos)",

            # Calls for elimination or exclusion
            r"\b(?:get\s+rid\s+of|eliminate|exterminate|deport\s+all|purge|cleanse)\s+"
            r"(?:them|these\s+people|those\s+people|all\s+\w+)",

            # Denial of humanity
            r"\b(?:they|these\s+people|those\s+people)\s+"
            r"(?:are\s+not|aren't)\s+(?:real\s+)?(?:people|humans|human)",

            # Concentrated derogatory language: "sick of" + group reference
            r"\bsick\s+of\s+(?:them|these|those|their|hearing\s+about\s+their)",
        ),
        severity=0.9,
    ),
    SafetyRule(
        name="self_harm",
        description="Detects encouragement or instructions for self-harm.",
        keywords=_SELF_HARM_KEYWORDS,
        severity=1.0,
    ),
    SafetyRule(
        name="illegal_activity",
        description="Detects instructions or encouragement for illegal acts.",
        keywords=_ILLEGAL_ACTIVITY_KEYWORDS,
        severity=0.8,
    ),
    SafetyRule(
        name="pii_patterns",
        description="Structural PII patterns (SSNs, credit cards, API keys).",
        patterns=(
            r"\b\d{3}-\d{2}-\d{4}\b",                        # SSN
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",     # Credit card
            r"\b(?:sk|pk|api|key)[_-][A-Za-z0-9_]{20,}\b",    # Prefixed API keys
            r"\bpassword\s*(?:[=:]|\bis\b)\s*\S{8,}",         # Bare passwords
        ),
        severity=0.8,
    ),
    SafetyRule(
        name="dangerous_synthesis",
        description="Detects step-by-step chemical or explosive synthesis "
                    "instructions involving known precursor substances.",
        patterns=(
            # Action verb + dangerous precursor chemical (within ~80 chars)
            # Catches: "mix potassium nitrate with ...", "dissolve the ammonium nitrate ..."
            r"(?:mix|combine|dissolve|heat|ignite|pour|add|stir|blend|grind)"
            r".{0,80}"
            r"(?:potassium nitrate|ammonium nitrate|sulfuric acid|nitric acid"
            r"|hydrochloric acid|sodium hydroxide|hydrogen peroxide"
            r"|acetone peroxide|red phosphorus|potassium chlorate"
            r"|potassium permanganate|sodium azide|mercury fulminate)",

            # Reverse: precursor chemical + action verb
            r"(?:potassium nitrate|ammonium nitrate|sulfuric acid|nitric acid"
            r"|hydrochloric acid|sodium hydroxide|hydrogen peroxide"
            r"|acetone peroxide|red phosphorus|potassium chlorate"
            r"|potassium permanganate|sodium azide|mercury fulminate)"
            r".{0,80}"
            r"(?:mix|combine|dissolve|heat|ignite|pour|add|stir|blend|grind)",

            # Named dangerous substances + synthesis/ratio language
            # Catches: "TATP is made by ...", "to synthesize ANFO, use a ratio ..."
            r"(?:tatp|anfo|rdx|hmtd|thermite|chlorine gas|mustard gas"
            r"|sarin|ricin|nerve agent)"
            r".{0,60}"
            r"(?:synthe|ratio|recipe|preparation|procedure|step\s*\d"
            r"|parts?\s+\w+\s+to\s+\d|percent|concentration)",

            # Synthesis/ratio language + named dangerous substances
            r"(?:synthe|ratio|recipe|preparation|procedure|step\s*\d"
            r"|parts?\s+\w+\s+to\s+\d|percent|concentration)"
            r".{0,60}"
            r"(?:tatp|anfo|rdx|hmtd|thermite|chlorine gas|mustard gas"
            r"|sarin|ricin|nerve agent)",

            # "how to make/synthesize/produce" + dangerous substance
            r"how\s+to\s+(?:make|synthesize|produce|create|manufacture|prepare)"
            r".{0,40}"
            r"(?:tatp|anfo|rdx|hmtd|thermite|chlorine gas|mustard gas"
            r"|sarin|ricin|nerve agent|acetone peroxide|mercury fulminate"
            r"|ammonium nitrate\s+fuel\s+oil)",

            # Mixing two dangerous reagents (acid + oxidizer combos)
            r"(?:sulfuric acid|nitric acid|hydrochloric acid)"
            r".{0,60}"
            r"(?:potassium nitrate|ammonium nitrate|potassium chlorate"
            r"|potassium permanganate|hydrogen peroxide|acetone)",
        ),
        severity=1.0,
    ),
)


# ---------------------------------------------------------------------------
# Entropy-based secret detection
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (bits per character) of string *s*."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _find_high_entropy_strings(
    text: str,
    *,
    min_length: int = 16,
    entropy_threshold: float = 4.0,
) -> list[tuple[int, int, str, float]]:
    """Find high-entropy substrings in *text* that may be secrets.

    Scans for contiguous non-whitespace tokens of at least *min_length*
    characters. Returns a list of (start, end, matched_text, entropy)
    tuples for tokens exceeding *entropy_threshold* bits per character.

    Parameters:
        text: The text to scan.
        min_length: Minimum token length to consider (short tokens
            have unreliable entropy). Default 16.
        entropy_threshold: Minimum Shannon entropy (bits/char) to flag.
            English text averages ~3.5; random secrets average ~5.5.
            Default 4.0 is a conservative threshold that catches most
            secrets while avoiding false positives on normal words.
    """
    results: list[tuple[int, int, str, float]] = []
    
    # Find contiguous non-whitespace sequences
    pattern = re.compile(rf"\S{{{min_length},}}")
    for match in pattern.finditer(text):
        token = match.group()
        start = match.start()
        end = match.end()
        
        # Strip common surrounding punctuation and backticks
        stripped_token = token.strip(".,\"'()[]{}~`!?;")
        
        if len(stripped_token) < min_length:
            continue
        
        # Exclude URLs — long non-whitespace strings but not secrets.
        if stripped_token.lower().startswith(("http://", "https://")):
            continue
            
        offset = token.find(stripped_token)
        adj_start = start + offset
        adj_end = adj_start + len(stripped_token)
        
        entropy = _shannon_entropy(stripped_token)
        if entropy > entropy_threshold:
            results.append((adj_start, adj_end, stripped_token, entropy))
            
    return results


_ENTROPY_RULE = SafetyRule(
    name="high_entropy_secret",
    description="Detects high-entropy strings that may be leaked secrets, "
                "API keys, passwords, tokens, or hashes.",
    severity=0.8,
)


# ---------------------------------------------------------------------------
# Protected content detection (n-gram overlap)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 4) -> set[str]:
    """Extract all character-level n-grams from *text*.

    Returns a set of unique n-grams. Case-insensitive to catch
    leaks regardless of casing changes.
    """
    lower = text.lower()
    return {lower[i:i + n] for i in range(len(lower) - n + 1)}


def _protected_content_overlap(
    output_text: str,
    protected_ngrams: frozenset[str],
    *,
    n: int = 4,
    threshold: float = 0.15,
) -> tuple[bool, float]:
    """Check if *output_text* overlaps significantly with protected content.

    Parameters:
        output_text: The LLM output to check.
        protected_ngrams: Pre-computed n-grams of the protected content.
        n: N-gram size (must match the size used to compute *protected_ngrams*).
        threshold: Minimum overlap ratio to flag as a leak.

    Returns:
        A tuple of (is_leak, overlap_ratio).
    """
    if not protected_ngrams:
        return False, 0.0

    output_ngrams = _char_ngrams(output_text, n)
    overlap = protected_ngrams & output_ngrams
    ratio = len(overlap) / len(protected_ngrams)
    return ratio >= threshold, ratio


_PROTECTED_CONTENT_RULE = SafetyRule(
    name="protected_content_leak",
    description="Detects output that reproduces substantial portions of "
                "the protected content (e.g. system prompt).",
    severity=0.9,
)

_DATA_ARTIFACT_RE = re.compile(
    r"\d{3,}"
    r"|https?://"
    r"|[@=]"
    r"|\$\d"
)

_MIN_SEGMENT_LENGTH = 20
_SEGMENT_OVERLAP_THRESHOLD = 0.55

_PROTECTED_SEGMENT_RULE = SafetyRule(
    name="protected_segment_leak",
    description="Detects output that reproduces a specific line or segment "
                "from the protected content (e.g. a single credential, "
                "employee record, or internal URL).",
    severity=0.9,
)


class SafetyPolicy:
    """Holds a set of :class:`SafetyRule` objects and evaluates text against them.

    Parameters:
        rules: Initial rules.  Defaults to a built-in set if *None*.
        use_defaults: If *True* (default), include the built-in rules.
        protected_content: Static reference text (e.g. system prompt) to
            detect leakage via n-gram overlap.  Empty string disables.
        protected_content_threshold: Minimum overlap ratio (0.0–1.0) to
            flag as a global protected content leak.  Default 0.20.
        protected_segment_threshold: Minimum overlap ratio (0.0–1.0) to
            flag an individual segment (line) leak.  Default 0.55.
    """

    def __init__(
        self,
        rules: list[SafetyRule] | None = None,
        *,
        use_defaults: bool = True,
        protected_content: str = "",
        protected_content_threshold: float = 0.20,
        protected_segment_threshold: float = _SEGMENT_OVERLAP_THRESHOLD,
    ) -> None:
        self._rules: list[SafetyRule] = []
        if use_defaults:
            self._rules.extend(_DEFAULT_RULES)
        if rules:
            self._rules.extend(rules)

        # Pre-compute protected content n-grams once.
        self._protected_content = protected_content
        self._protected_ngrams: frozenset[str] = (
            frozenset(_char_ngrams(protected_content))
            if protected_content else frozenset()
        )
        self._protected_content_threshold = protected_content_threshold

        # Segment-level n-grams for fine-grained leak detection.
        self._protected_segment_threshold = protected_segment_threshold
        self._protected_segments: tuple[tuple[str, frozenset[str]], ...] = ()
        if protected_content:
            segments = []
            for line in protected_content.splitlines():
                stripped = line.strip()
                if len(stripped) >= _MIN_SEGMENT_LENGTH and _DATA_ARTIFACT_RE.search(stripped):
                    ngrams = frozenset(_char_ngrams(stripped))
                    if ngrams:
                        segments.append((stripped, ngrams))
            self._protected_segments = tuple(segments)

    # -- public API ---------------------------------------------------------

    @property
    def rules(self) -> tuple[SafetyRule, ...]:
        """Return an immutable snapshot of the current rules."""
        return tuple(self._rules)

    @property
    def protected_content_text(self) -> str:
        """Return the raw protected content string for semantic evaluation."""
        return self._protected_content

    def add_rule(self, rule: SafetyRule) -> None:
        """Append a custom rule to the policy.

        .. warning::

            This method is **not thread-safe**.  In a multi-threaded context,
            calling ``add_rule`` while another thread is inside ``evaluate``
            may produce inconsistent results.  For concurrent use, build the
            full rule list before constructing the policy.
        """
        self._rules.append(rule)

    def evaluate(self, text: str, *, is_refusal: bool = False) -> list[RuleMatch]:
        """Scan *text* against every rule and return all matches.

        Parameters:
            text: The text to evaluate.
            is_refusal: When *True*, keyword matching is skipped.
                Pattern matching and entropy scanning always run regardless.

        Matching strategy per keyword depends on the rule's ``match_mode``:

        * ``"boundary"`` (default): single-word keywords use ``\\b`` word
          boundaries; multi-word keywords (containing a space) fall back to
          substring matching.
        * ``"substring"``: all keywords use plain substring matching.

        Regex patterns in ``rule.patterns`` are always matched via ``re``
        with ``IGNORECASE``.
        """
        matches: list[RuleMatch] = []
        for rule in self._rules:
            # Keywords are topic detectors — suppressed during refusals.
            if not is_refusal:
                matches.extend(self._match_keywords(rule, text))
            # Patterns are artifact detectors — always run.
            matches.extend(self._match_patterns(rule, text))
        
        # Entropy is an artifact detector — always runs.
        matches.extend(self._match_entropy(text))

        # Protected content detection — compute output n-grams once.
        if self._protected_ngrams or self._protected_segments:
            output_ngrams = _char_ngrams(text)
            matches.extend(self._match_protected_content(text, output_ngrams))
            matches.extend(self._match_protected_segments(text, output_ngrams))
        
        return matches

    def evaluate_prompt(self, prompt: str) -> list[RuleMatch]:
        """Run keyword and pattern checks on the user prompt.

        Only checks keywords and patterns — entropy and protected content
        detection are not relevant for prompts.  This is used as a weak
        signal: a prompt that triggers safety keywords indicates the
        interaction topic is sensitive.

        Parameters:
            prompt: The user's input prompt.

        Returns:
            List of rule matches found in the prompt.
        """
        matches: list[RuleMatch] = []
        for rule in self._rules:
            matches.extend(self._match_keywords(rule, prompt))
            matches.extend(self._match_patterns(rule, prompt))
        return matches

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _match_keywords(rule: SafetyRule, text: str) -> Iterator[RuleMatch]:
        """Yield :class:`RuleMatch` for every keyword hit."""
        for keyword in rule.keywords:
            use_boundary = (
                rule.match_mode == "boundary" and " " not in keyword
            )
            if use_boundary:
                pattern = re.compile(
                    r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE
                )
                for m in pattern.finditer(text):
                    yield RuleMatch(
                        rule=rule,
                        matched_keyword=keyword,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        match_type="keyword",
                    )
            else:
                # Substring matching (multi-word phrases or explicit mode).
                lower_text = text.lower()
                lower_kw = keyword.lower()
                kw_len = len(lower_kw)
                start = 0
                while True:
                    idx = lower_text.find(lower_kw, start)
                    if idx == -1:
                        break
                    end = idx + kw_len
                    yield RuleMatch(
                        rule=rule,
                        matched_keyword=keyword,
                        matched_text=text[idx:end],
                        start=idx,
                        end=end,
                        match_type="keyword",
                    )
                    start = end

    @staticmethod
    def _match_patterns(rule: SafetyRule, text: str) -> Iterator[RuleMatch]:
        """Yield :class:`RuleMatch` for every regex pattern hit.

        Uses pre-compiled patterns from ``rule.compiled_patterns``
        (compiled once with ``re.IGNORECASE`` at rule construction time).
        """
        for compiled, pat_str in zip(rule.compiled_patterns, rule.patterns):
            for m in compiled.finditer(text):
                yield RuleMatch(
                    rule=rule,
                    matched_keyword=pat_str,
                    matched_text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    match_type="pattern",
                )

    @staticmethod
    def _match_entropy(text: str) -> Iterator[RuleMatch]:
        """Yield :class:`RuleMatch` for high-entropy substrings (potential secrets)."""
        seen = set()
        for start, end, matched_text, entropy in _find_high_entropy_strings(text):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            yield RuleMatch(
                rule=_ENTROPY_RULE,
                matched_keyword=f"entropy:{entropy:.2f}",
                matched_text=matched_text,
                start=start,
                end=end,
                match_type="entropy",
            )

    def _match_protected_content(self, text: str, output_ngrams: set[str]) -> Iterator[RuleMatch]:
        """Yield a :class:`RuleMatch` if the output overlaps significantly with protected content."""
        if not self._protected_ngrams:
            return

        overlap = self._protected_ngrams & output_ngrams
        ratio = len(overlap) / len(self._protected_ngrams)

        if ratio >= self._protected_content_threshold:
            yield RuleMatch(
                rule=_PROTECTED_CONTENT_RULE,
                matched_keyword=f"overlap:{ratio:.2%}",
                matched_text="",
                start=0,
                end=len(text),
                match_type="protected_content",
            )

    def _match_protected_segments(
        self, text: str, output_ngrams: set[str],
    ) -> Iterator[RuleMatch]:
        """Yield a :class:`RuleMatch` for each protected segment that
        substantially appears in the output.

        Unlike the global overlap check, this catches partial leaks —
        even a single leaked credential line is flagged regardless of
        how short the output is relative to the full protected content.
        """
        if not self._protected_segments:
            return

        for segment_text, segment_ngrams in self._protected_segments:
            overlap = segment_ngrams & output_ngrams
            ratio = len(overlap) / len(segment_ngrams)
            if ratio >= self._protected_segment_threshold:
                yield RuleMatch(
                    rule=_PROTECTED_SEGMENT_RULE,
                    matched_keyword=f"segment_overlap:{ratio:.2%}",
                    matched_text=segment_text[:80],
                    start=0,
                    end=len(text),
                    match_type="protected_segment",
                )
