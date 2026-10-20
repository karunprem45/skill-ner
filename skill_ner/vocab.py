"""Skill patterns, loaded from SkillRadar so both projects share one vocabulary.

VOCAB is read from SkillRadar's src/extract_rules.py with ast.literal_eval instead
of being imported, because importing that module also opens SkillRadar's database.
The two custom patterns (R and C++) mirror the ones defined in that file.
"""

import ast
import re
from pathlib import Path


def load_vocab(extract_rules_path):
    tree = ast.parse(Path(extract_rules_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "VOCAB":
            return ast.literal_eval(node.value)
    raise ValueError(f"No VOCAB dict found in {extract_rules_path}")


def build_patterns(vocab):
    patterns = {
        skill: re.compile(r"\b(?:" + "|".join(forms) + r")\b", re.IGNORECASE)
        for skill, forms in vocab.items()
        if forms
    }
    # Same special cases as SkillRadar: "R" needs context, and \b can't follow "+"
    patterns["R"] = re.compile(r"(?:\bR programming\b|[,/(]\s*R\s*[,/)]|\bin R\b)")
    patterns["C++"] = re.compile(r"\bc\+\+", re.IGNORECASE)
    return patterns


def canonical_skill(entity_text, patterns):
    """Map an entity string to a vocabulary skill, or None if it is out of vocabulary."""
    if entity_text.strip() == "R":
        return "R"
    for skill, pattern in patterns.items():
        if pattern.search(entity_text):
            return skill
    return None
