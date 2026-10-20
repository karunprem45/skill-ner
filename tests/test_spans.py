import re

from skill_ner.spans import regex_spans, resolve_overlaps, window
from skill_ner.vocab import build_patterns, canonical_skill

VOCAB = {
    "Machine Learning": ["machine learning", r"\bml\b"],
    "Java": ["java(?!script)"],
    "JavaScript": ["javascript"],
    "Python": ["python"],
    "R": [],
    "C++": [],
}
PATTERNS = build_patterns(VOCAB)


def spans_text(text):
    return [text[s:e] for s, e in resolve_overlaps(regex_spans(text, PATTERNS))]


def test_lookahead_keeps_java_and_javascript_apart():
    assert spans_text("Java and JavaScript") == ["Java", "JavaScript"]


def test_r_span_is_only_the_letter():
    text = "Experience in R, Python"
    assert spans_text(text) == ["R", "Python"]


def test_cpp_matches_without_trailing_boundary():
    assert spans_text("Strong C++ skills") == ["C++"]


def test_overlap_keeps_longest_span():
    assert resolve_overlaps([(0, 7), (0, 16), (8, 16)]) == [(0, 16)]


def test_overlap_keeps_disjoint_spans():
    assert resolve_overlaps([(10, 15), (0, 5)]) == [(0, 5), (10, 15)]


def test_canonical_skill_maps_and_flags_oov():
    assert canonical_skill("machine learning", PATTERNS) == "Machine Learning"
    assert canonical_skill("R", PATTERNS) == "R"
    assert canonical_skill("dbt", PATTERNS) is None


def test_window_collapses_whitespace_and_truncates():
    text = window("Title", "a   b\n\nc" + " x" * 5000, limit=10)
    assert text.startswith("Title\na b c")
    assert len(text) == len("Title\n") + 10
    assert not re.search(r"\s{2,}", text)
