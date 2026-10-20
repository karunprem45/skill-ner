"""Build spaCy training data from SkillRadar postings, labeled by its regex extractor.

These are silver labels: the regex decides where each SKILL entity is. The 15
hand-labeled gold postings are left out entirely so evaluate.py can score the
model on postings it has never seen.

Run:  python build_data.py --skillradar ~/Desktop/GITHUB_PORT/live/skillradar
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

import spacy
from spacy.tokens import DocBin

from skill_ner.spans import regex_spans, resolve_overlaps, window
from skill_ner.vocab import build_patterns, load_vocab

SEED = 42


def load_gold_ids(skillradar):
    gold = json.loads((skillradar / "data" / "gold_set.json").read_text())
    return {k for k in gold if not k.startswith("_")}


def shuffled_postings(skillradar):
    """Every non-gold posting with a description, in a fixed seeded order."""
    gold_ids = load_gold_ids(skillradar)
    with sqlite3.connect(skillradar / "data" / "skillradar.db") as conn:
        rows = conn.execute("SELECT source_id, title, description FROM jobs WHERE description != ''").fetchall()
    rows = sorted(r for r in rows if r[0] not in gold_ids)
    random.Random(SEED).shuffle(rows)
    return rows


def select_training_rows(rows, patterns, n):
    """The first n postings (in shuffled order) that have at least one regex match."""
    selected = []
    for source_id, title, description in rows:
        if len(selected) == n:
            break
        text = window(title, description)
        spans = resolve_overlaps(regex_spans(text, patterns))
        if spans:
            selected.append((source_id, text, spans))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillradar", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4000, help="postings to sample")
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    args = parser.parse_args()

    patterns = build_patterns(load_vocab(args.skillradar / "src" / "extract_rules.py"))
    rows = shuffled_postings(args.skillradar)
    selected = select_training_rows(rows, patterns, args.n)

    nlp = spacy.blank("en")
    docs, n_spans, n_misaligned = [], 0, 0
    for source_id, text, spans in selected:
        doc = nlp.make_doc(text)
        ents = []
        for start, end in spans:
            span = doc.char_span(start, end, label="SKILL", alignment_mode="contract")
            if span is None:
                n_misaligned += 1
            else:
                ents.append(span)
        doc.ents = ents
        n_spans += len(ents)
        docs.append(doc)

    split = int(len(docs) * 0.9)
    args.out.mkdir(exist_ok=True)
    DocBin(docs=docs[:split]).to_disk(args.out / "train.spacy")
    DocBin(docs=docs[split:]).to_disk(args.out / "dev.spacy")

    stats = {
        "postings_available": len(rows),
        "gold_postings_excluded": len(load_gold_ids(args.skillradar)),
        "train_docs": split,
        "dev_docs": len(docs) - split,
        "skill_spans": n_spans,
        "spans_dropped_misaligned": n_misaligned,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "train_ids.json").write_text(json.dumps([sid for sid, _, _ in selected]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
