"""Score the spaCy NER model against SkillRadar's regex extractor on the gold set.

Both extractors are scored the same way SkillRadar's src/evaluate.py scores the
regex: per posting, the set of vocabulary skills found vs. the hand-labeled set.
An NER entity counts toward a vocabulary skill when the skill's own regex
matches the entity text. Entities that match no skill are out of vocabulary:
they can't be scored against the gold labels, so they are reported separately
as candidate new skills, counted over postings the model never trained on.

Run:  python evaluate.py --skillradar ~/Desktop/GITHUB_PORT/live/skillradar
"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import spacy

from build_data import load_gold_ids, shuffled_postings
from skill_ner.spans import window
from skill_ner.vocab import build_patterns, canonical_skill, load_vocab

N_UNSEEN = 1000          # unseen postings scanned for out-of-vocabulary candidates
MIN_OOV_COUNT = 1        # report every out-of-vocabulary entity string


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillradar", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/model-best"))
    args = parser.parse_args()

    patterns = build_patterns(load_vocab(args.skillradar / "src" / "extract_rules.py"))
    gold = json.loads((args.skillradar / "data" / "gold_set.json").read_text())
    gold = {k: set(v) for k, v in gold.items() if not k.startswith("_")}
    nlp = spacy.load(args.model)

    with sqlite3.connect(args.skillradar / "data" / "skillradar.db") as conn:
        placeholders = ",".join("?" * len(gold))
        jobs = {sid: (title, desc) for sid, title, desc in conn.execute(
            f"SELECT source_id, title, description FROM jobs WHERE source_id IN ({placeholders})", list(gold))}

    counts = {"regex": [0, 0, 0], "ner": [0, 0, 0]}
    errors = []
    for source_id, truth in gold.items():
        text = window(*jobs[source_id])
        regex_pred = {skill for skill, pattern in patterns.items() if pattern.search(text)}
        ner_pred = {canonical_skill(ent.text, patterns) for ent in nlp(text).ents} - {None}
        for name, pred in [("regex", regex_pred), ("ner", ner_pred)]:
            counts[name][0] += len(pred & truth)
            counts[name][1] += len(pred - truth)
            counts[name][2] += len(truth - pred)
        if ner_pred != truth:
            errors.append({"source_id": source_id, "ner_missed": sorted(truth - ner_pred),
                           "ner_spurious": sorted(ner_pred - truth)})

    # Out-of-vocabulary candidates, from postings the model never saw. Training took
    # the first postings (in shuffled order) that had a regex match, so the skipped
    # postings before the cutoff are biased toward having no skills at all. Taking
    # postings after the last training posting gives an unbiased random sample.
    rows = shuffled_postings(args.skillradar)
    last_train_id = json.loads(Path("corpus/train_ids.json").read_text())[-1]
    cutoff = next(i for i, r in enumerate(rows) if r[0] == last_train_id)
    unseen = rows[cutoff + 1:cutoff + 1 + N_UNSEEN]
    oov, n_entities = Counter(), 0
    for doc in nlp.pipe((window(title, desc) for _, title, desc in unseen), batch_size=64):
        n_entities += len(doc.ents)
        oov.update(ent.text.strip().lower() for ent in doc.ents if canonical_skill(ent.text, patterns) is None)
    candidates = [(term, n) for term, n in oov.most_common() if n >= MIN_OOV_COUNT]

    results = {
        "gold_postings": len(gold),
        "gold_labels": sum(len(v) for v in gold.values()),
        "regex": prf(*counts["regex"]),
        "spacy_ner": prf(*counts["ner"]),
        "unseen_postings_scanned": len(unseen),
        "unseen_entities": n_entities,
        "unseen_oov_entities": sum(oov.values()),
        "unseen_oov_share": round(sum(oov.values()) / n_entities, 4) if n_entities else 0.0,
        "oov_candidates_min_count": MIN_OOV_COUNT,
        "oov_candidates": len(candidates),
        "oov_top": candidates[:25],
        "ner_errors": errors,
    }
    Path("results.json").write_text(json.dumps(results, indent=2))

    print(f"Gold set: {results['gold_postings']} postings, {results['gold_labels']} labels")
    for name in ("regex", "spacy_ner"):
        m = results[name]
        print(f"  {name:10} precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
    print(f"\n{n_entities} entities in {len(unseen)} unseen postings; "
          f"{sum(oov.values())} ({results['unseen_oov_share']:.1%}) are out of vocabulary. Top 25:")
    for term, n in candidates[:25]:
        print(f"  {n:4}  {term}")


if __name__ == "__main__":
    main()
