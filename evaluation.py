from collections import Counter
from typing import Iterable


def normalize_for_rouge(text: str, remove_spaces: bool = False) -> str:
    normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.split())
    if remove_spaces:
        normalized = normalized.replace(" ", "")
    return normalized


def char_tokenize(text: str) -> list[str]:
    return list(text)


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _overlap_count(candidate: Counter[tuple[str, ...]], reference: Counter[tuple[str, ...]]) -> int:
    overlap = 0
    for ngram, count in candidate.items():
        overlap += min(count, reference.get(ngram, 0))
    return overlap


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_n(tokens_candidate: list[str], tokens_reference: list[str], n: int) -> dict[str, float]:
    cand_counts = _ngram_counts(tokens_candidate, n)
    ref_counts = _ngram_counts(tokens_reference, n)
    overlap = _overlap_count(cand_counts, ref_counts)
    precision = _safe_divide(overlap, sum(cand_counts.values()))
    recall = _safe_divide(overlap, sum(ref_counts.values()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _lcs_length(tokens_candidate: list[str], tokens_reference: list[str]) -> int:
    if not tokens_candidate or not tokens_reference:
        return 0

    prev = [0] * (len(tokens_reference) + 1)
    for cand in tokens_candidate:
        curr = [0]
        for j, ref in enumerate(tokens_reference, start=1):
            if cand == ref:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l(tokens_candidate: list[str], tokens_reference: list[str]) -> dict[str, float]:
    lcs_len = _lcs_length(tokens_candidate, tokens_reference)
    precision = _safe_divide(lcs_len, len(tokens_candidate))
    recall = _safe_divide(lcs_len, len(tokens_reference))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def compute_rouge(candidate: str, reference: str) -> dict[str, dict[str, float]]:
    normalized_candidate = normalize_for_rouge(candidate, remove_spaces=True)
    normalized_reference = normalize_for_rouge(reference, remove_spaces=True)

    tokens_candidate = char_tokenize(normalized_candidate)
    tokens_reference = char_tokenize(normalized_reference)

    return {
        "rouge1": _rouge_n(tokens_candidate, tokens_reference, 1),
        "rouge2": _rouge_n(tokens_candidate, tokens_reference, 2),
        "rougeL": _rouge_l(tokens_candidate, tokens_reference),
    }
