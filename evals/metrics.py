"""Evaluation metrics for the Bangla benchmark (pure stdlib).

- ``exact_match`` — token-normalized exact match against a reference.
- ``rouge`` — ROUGE-1/2/L F1 (character-level tokens).
- ``language_detection`` — majority-script classifier (Bengali/English).
- ``bengali_script_ratio`` — fraction of Bengali-script codepoints.
- ``mean_ci`` — mean with a 95% confidence interval (normal approximation).
"""

from __future__ import annotations

import math
import statistics
import unicodedata

__all__ = [
    "bengali_script_ratio",
    "context_token_coverage",
    "context_unsupported_token_rate",
    "exact_match",
    "language_detection",
    "mean_ci",
    "normalize_answer",
    "rouge",
    "semantic_entailment",
    "semantic_contradiction",
    "preference_judge",
    "refusal_classifier",
    "helpfulness_metrics",
    "bangla_quality_metrics",
]

_BENGALI_RANGES = ((0x0980, 0x09FF), (0x0964, 0x0965))
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A))


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def bengali_script_ratio(text: str) -> float:
    """Fraction of Bengali/Latin script chars that are Bengali (vowel signs count)."""
    bn = sum(_in_ranges(ord(ch), _BENGALI_RANGES) for ch in text)
    la = sum(_in_ranges(ord(ch), _LATIN_RANGES) for ch in text)
    total = bn + la
    if total == 0:
        return 0.0
    return bn / total


def language_detection(text: str) -> str:
    """Return ``'bn'`` or ``'en'`` based on the dominant script character count."""
    bn = sum(_in_ranges(ord(ch), _BENGALI_RANGES) for ch in text)
    la = sum(_in_ranges(ord(ch), _LATIN_RANGES) for ch in text)
    if bn > la:
        return "bn"
    if la > bn:
        return "en"
    return "bn" if bn else "en"


def normalize_answer(text: str) -> str:
    """Lowercase, NFC-normalize, and collapse whitespace for comparison."""
    text = unicodedata.normalize("NFC", text or "")
    text = " ".join(text.lower().split())
    return text.strip(' .।,;:!?()[]{}"')


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in normalize_answer(text):
        if unicodedata.category(char)[0] in {"L", "M", "N"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _context_token_counts(answer: str, context: str) -> tuple[int, int]:
    answer_tokens = _lexical_tokens(answer)
    context_tokens = set(_lexical_tokens(context))
    matched = sum(token in context_tokens for token in answer_tokens)
    return matched, len(answer_tokens)


def context_token_coverage(answer: str, context: str) -> float:
    """Fraction of answer tokens also present in context (lexical baseline only)."""
    matched, total = _context_token_counts(answer, context)
    return matched / total if total else 0.0


def context_unsupported_token_rate(answer: str, context: str) -> float:
    """Fraction of answer tokens absent from context (lexical baseline only)."""
    matched, total = _context_token_counts(answer, context)
    return (total - matched) / total if total else 0.0


def _lcs(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def _rouge_f1(prediction: str, reference: str, n: int | None) -> float:
    def tokens(text: str) -> list[str]:
        if n == 1 or n is None:
            return text.split()
        grams: list[str] = []
        parts = text.split()
        for i in range(len(parts) - n + 1):
            grams.append(" ".join(parts[i : i + n]))
        return grams

    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return 0.0
    if n is None:
        overlap = _lcs(pred, ref)
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(ref) if ref else 0.0
    else:
        pred_g, ref_g = tokens(prediction), tokens(reference)
        if not pred_g or not ref_g:
            return 0.0
        precision = sum(1 for g in pred_g if g in set(ref_g)) / len(pred_g)
        recall = sum(1 for g in ref_g if g in set(pred_g)) / len(ref_g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 scores."""
    return {
        "rouge1": _rouge_f1(prediction, reference, 1),
        "rouge2": _rouge_f1(prediction, reference, 2),
        "rougeL": _rouge_f1(prediction, reference, None),
    }


def mean_ci(values: list[float], z: float = 1.96) -> dict[str, float]:
    """Mean, std, and a normal-approximation 95% CI."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - z * se, "ci_high": mean + z * se}


# --- WS-5 semantic consistency (lexical NLI proxy, stdlib only) ---

_REFUSAL_PATTERNS = (
    "না করতে পারব",
    "পারব না",
    "দিতে পারব না",
    "সাহায্য করতে পারব না",
    "করতে পারব না",
    "i cannot",
    "i can't",
    "i'm not able",
    "i am not able",
    "sorry, i can't",
    "sorry i can't",
    "i cannot",
    "i can not",
    "i'm sorry",
    "as an ai",
    "as a language model",
    "not appropriate",
    "not safe",
    "not ethical",
    "against my",
    "violates",
    "unable to",
    "won't be able to",
    "not possible",
    "can't help",
    "cannot help",
    "refuse",
    "refusing",
    "refusal",
    "তৈরি করতে পারব না",
    "লেখা সম্ভব নয়",
    "সম্ভব নয়",
    "মনে হচ্ছে না",
    "নিধ্বংস করা উচিত নয়",
    "অনৈতিক",
)

# Signals that a refusal is *soft* (declines + offers an alternative) rather
# than a hard refusal. Used to classify "partial". These are explicit
# "offering an answer" markers — a bare "but" connecting clauses is not enough.
_PARTIAL_SIGNALS = (
    "but here",
    "but here's",
    "but here is",
    "কিন্তু এটার সমাধান",
    "কিন্তু এই",
    "কিন্তু তা হলে",
    "instead, here",
    "instead, the answer",
    "but the answer is",
    "but you can",
    "but you could",
)


def semantic_entailment(answer: str, context: str) -> float:
    """Lexical NLI proxy: fraction of answer content units present in context.

    A content unit is a normalized 1- or 2-gram. This is a diagnostic baseline,
    not a true semantic entailment model; pair with ``semantic_contradiction``
    to flag unsupported claims.
    """
    answer_units = _content_units(answer)
    context_units = set(_content_units(context))
    if not answer_units:
        return 0.0
    matched = sum(1 for unit in answer_units if unit in context_units)
    return matched / len(answer_units)


def semantic_contradiction(answer: str, context: str) -> float:
    """Fraction of answer content units absent from context (unsupported claim rate).

    Higher means more of the answer invents facts not grounded in context.
    """
    return 1.0 - semantic_entailment(answer, context)


def _content_units(text: str) -> list[str]:
    """Normalized unigrams and bigrams over lexical tokens (Bengali + Latin)."""
    tokens = _lexical_tokens(text)
    units = list(tokens)
    for i in range(len(tokens) - 1):
        units.append(f"{tokens[i]} {tokens[i + 1]}")
    return units


# --- WS-4 preference judging (lexical overlap proxy) ---

def preference_judge(chosen: str, rejected: str, reference: str) -> dict[str, float]:
    """Rule-based preference judge: does ``chosen`` better match ``reference``?

    Returns overlap-based scores for both candidates plus a win signal. This is
    a deterministic baseline; real judging uses a reward model.
    """
    chosen_score = _overlap(chosen, reference)
    rejected_score = _overlap(rejected, reference)
    return {
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "win": 1.0 if chosen_score > rejected_score else 0.0,
        "tie": 1.0 if chosen_score == rejected_score else 0.0,
    }


def _overlap(candidate: str, reference: str) -> float:
    cand = set(_content_units(candidate))
    ref = set(_content_units(reference))
    if not cand or not ref:
        return 0.0
    return len(cand & ref) / len(cand)


# --- WS-6/WS-9 refusal classifier ---

def refusal_classifier(text: str) -> str:
    """Three-way refusal classifier: ``answer | refuse | partial``.

    ``partial`` is a soft refusal: the response acknowledges the request but
    declines to fulfil it while offering an alternative (e.g. "I can't do X,
    but here's Y"). A hard refusal declines outright.
    """
    if not text or not text.strip():
        return "answer"
    lowered = text.strip().lower()
    if not any(pat in lowered for pat in _REFUSAL_PATTERNS):
        return "answer"
    if any(sig in lowered for sig in _PARTIAL_SIGNALS):
        return "partial"
    return "refuse"


def helpfulness_metrics(
    response: str, reference: str = "", expected_tokens: int | None = None
) -> dict[str, float]:
    """Return deterministic helpfulness proxies, not a human preference judgment.

    Reference coverage and verbosity fit are lexical baselines. When neither a
    reference nor an expected length is available, the aggregate uses only
    answerability and non-refusal.
    """
    response_tokens = _lexical_tokens(response)
    reference_tokens = _lexical_tokens(reference)
    answerability = float(bool(response_tokens))
    non_refusal = float(refusal_classifier(response) == "answer")
    information_sufficiency = context_token_coverage(reference, response)

    target_length = expected_tokens or len(reference_tokens)
    verbosity_fit = (
        min(len(response_tokens), target_length) / max(len(response_tokens), target_length)
        if target_length > 0 and response_tokens
        else 0.0
    )
    components = [answerability, non_refusal]
    if reference_tokens:
        components.extend((information_sufficiency, verbosity_fit))
    elif expected_tokens:
        components.append(verbosity_fit)

    return {
        "answerability": answerability,
        "information_sufficiency": information_sufficiency,
        "verbosity_fit": verbosity_fit,
        "non_refusal": non_refusal,
        "helpfulness_score": statistics.mean(components),
    }


def bangla_quality_metrics(response: str, reference: str = "") -> dict[str, float]:
    """Return script, language, and reference-similarity proxies for Bangla.

    These deterministic signals do not assess grammar, register, or
    transliteration fidelity; those require dedicated linguistic evaluation.
    """
    script_ratio = bengali_script_ratio(response)
    language_correct = float(
        bool(_lexical_tokens(response)) and language_detection(response) == "bn"
    )
    reference_similarity = rouge(response, reference)["rougeL"] if reference else 0.0
    components = [script_ratio, language_correct]
    if reference:
        components.append(reference_similarity)

    return {
        "bengali_script_ratio": script_ratio,
        "language_correct": language_correct,
        "reference_similarity": reference_similarity,
        "bangla_quality_score": statistics.mean(components),
    }
