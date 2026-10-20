"""Turn regex matches into non-overlapping character spans for NER training."""


def regex_spans(text, patterns):
    """All (start, end) matches across every pattern, before overlap resolution."""
    spans = []
    for skill, pattern in patterns.items():
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if skill == "R":
                # The R pattern matches context ("in R", ", R,"); keep only the letter itself
                start = m.start() + m.group().index("R")
                spans.append((start, start + 1))
                continue
            spans.append((start, end))
    return spans


def resolve_overlaps(spans):
    """Keep the longest span when two overlap; ties go to the earlier one.

    spaCy can't train on overlapping entities, and "machine learning" should win
    over a shorter match inside it.
    """
    chosen = []
    for start, end in sorted(set(spans), key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(end <= s or start >= e for s, e in chosen):
            chosen.append((start, end))
    return sorted(chosen)


def window(title, description, limit=4200):
    """Same text window SkillRadar's evaluator scores against."""
    return f"{title}\n{' '.join((description or '').split())[:limit]}"
