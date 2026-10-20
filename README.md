# Skill NER: spaCy entity recognition trained on regex silver labels

Can a spaCy NER model learn to find skills in job postings from labels that a regex
generated, and does it find anything the regex misses?

The data comes from [SkillRadar](https://github.com/karunprem45/skillradar), which
collects US job postings from 4 job-site APIs. SkillRadar's rule-based extractor
matches a 91-skill vocabulary with word-boundary and lookahead regexes. Here, that
extractor labels the training data, and a spaCy NER model learns from it.

## How it works

1. **Silver labels** (`build_data.py`). Every regex match becomes a `SKILL` character
   span. Overlapping matches keep the longest span, so "machine learning" wins over
   "ml". The R pattern needs context such as "in R", so its span is trimmed to the
   letter itself. 4,000 postings are sampled (seed 42), giving 14,592 spans; 42 spans
   (0.3%) are dropped because they don't line up with token boundaries.
2. **Training**. A blank English spaCy pipeline with one NER component
   (`configs/config.cfg`, CPU efficiency preset, 3,000 steps), on 3,600 training and
   400 dev postings.
3. **Evaluation** (`evaluate.py`). Both extractors are scored on SkillRadar's
   hand-labeled gold set: 15 postings and 110 skill labels, none of them used in
   training. Scoring works the same way as SkillRadar's own evaluator: per posting,
   the set of vocabulary skills found vs. the gold set.

## Results

| Extractor | Precision | Recall | F1 |
|---|---|---|---|
| Regex (the teacher) | 0.991 | 1.000 | 0.995 |
| spaCy NER (the student) | 0.991 | 0.955 | 0.972 |

The model reaches 0.997 F1 on the dev set, but those labels came from the regex, so
that only measures how well it copies the regex. On the gold set it does slightly
worse than the regex. All 5 of its misses are multi-word or rare skills: Model
Monitoring, MLOps, Data Modeling, Databricks, and Time Series.

**Does it find skills outside the vocabulary?** Barely. On 1,000 random postings the
model never saw, it found 2,023 entities. Only 12 of them (0.6%) map to no
vocabulary skill, and most of those are noise ("solicit", "Firefox"). A few are
plausible skills: "statistical inference", "natural language querying", and
"vector retrieval".

**Takeaway.** Trained on silver labels, the model learns its teacher's vocabulary and
little beyond it. Finding new skills would need hand-annotated spans that include
skills the regex doesn't know, which is the job annotation tools like Prodigy are
built for.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_data.py --skillradar ../GITHUB_PORT/live/skillradar
python -m spacy train configs/config.cfg --output models --paths.train corpus/train.spacy --paths.dev corpus/dev.spacy
python evaluate.py --skillradar ../GITHUB_PORT/live/skillradar
pytest -q
```

`--skillradar` points to a local clone of SkillRadar. The vocabulary is read from its
`src/extract_rules.py`, so the two projects always use the same skill list.
`results.json` holds the full output of `evaluate.py`.
