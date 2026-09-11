# NimbusAir Support Agent (Hiver take-home)

An AI customer-support agent for one brand (`NimbusAir`, synthetic stand-in
below — real submission targets a real brand from the Kaggle Twitter support
dataset) that classifies incoming messages, drafts a reply grounded in that
brand's own historical resolutions, and decides auto-handle vs. escalate with
a stated reason.

**Full report:** [`report/report.md`](report/report.md)
**Decision log:** [`decision_log.md`](decision_log.md)

## Reproduce in under 15 minutes

```bash
pip install -r requirements.txt

# 1. Generate the offline synthetic dataset (schema-identical to the real
#    Kaggle twcs.csv). See "Using the real dataset" below to swap this out.
python scripts/build_sample_data.py

# 2. Build the demo golden set (uses embedded ground truth from step 1 --
#    see eval/build_sample_golden_set.py docstring for why this isn't the
#    real hand-labeling process)
python -m eval.build_sample_golden_set

# 3. Run one message through the agent
python -m src.pipeline --brand NimbusAir --message "My flight was delayed 6 hours and nobody told me"

# 4. Run the full eval harness: system vs. trivial baseline vs. simple
#    baseline, intent metrics, escalation metrics, reply-quality judge,
#    judge-vs-human agreement
python -m eval.run_eval --brand NimbusAir --golden eval/sample_golden_set.csv
```

Everything above runs **offline, with no API key**, using deterministic
rule-based classification/reply-drafting/judging fallbacks. This was a
deliberate choice — see [decision_log.md](decision_log.md) #2 — so a grader
can reproduce headline numbers without provisioning an LLM API key.

## Using an LLM (recommended for real evaluation)

```bash
export ANTHROPIC_API_KEY=sk-...
python -m src.pipeline --brand NimbusAir --message "..."   # now uses LLM classify + LLM reply draft
python -m eval.run_eval --brand NimbusAir --golden eval/sample_golden_set.csv  # now uses LLM judge
```

The classifier, reply generator, and judge all auto-detect the key and
upgrade from rule-based/heuristic to LLM-backed. Every module degrades
gracefully back to rule-based on any API error, so a flaky network never
crashes the pipeline mid-eval.

## Using the real Kaggle dataset

1. Download `twcs.csv` from [Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
2. Place it at `data/twcs.csv` (overwrites the synthetic file)
3. Pick a real brand handle present in the data (e.g. `AmazonHelp`,
   `AppleSupport`, `Uber_Support`, `SpotifyCares`) and pass `--brand <handle>`
   to every script below instead of `NimbusAir`
4. Build a REAL golden set (this is the actual deliverable, not the demo one):
   ```bash
   python -m eval.build_golden_set --brand AmazonHelp --n 200 --out eval/golden_set_unlabeled.csv
   # hand-label gold_intent / gold_escalate / gold_escalate_reason per eval/label_schema.md
   # save as eval/golden_set.csv
   python -m eval.run_eval --brand AmazonHelp --golden eval/golden_set.csv
   ```
5. Optionally run `python -m scripts.derive_intents --brand AmazonHelp` first
   to sanity-check whether the 6 intents in `src/config.py:INTENTS` fit the
   real brand's traffic, or need adjusting (see decision_log.md #4).

## Repo layout

```
src/
  config.py          brand, intents, keyword rules, escalation thresholds
  data_loader.py      loads twcs.csv, filters to one brand, builds resolved (customer->agent) pairs
  retrieval.py         TF-IDF retrieval of similar historically-resolved threads
  classifier.py        intent classification (LLM zero-shot / keyword-rule fallback)
  reply_generator.py   drafts a reply grounded in retrieved examples (LLM / template fallback)
  escalation.py        auto-handle vs escalate decision, with a stated reason
  pipeline.py           wires the above into SupportAgent.run(message) -> AgentOutput
  llm_client.py          thin Anthropic API wrapper
baselines/
  trivial_baseline.py   majority-class intent, canned reply, never escalates
  simple_baseline.py     TF-IDF + LogisticRegression intent classifier, per-intent canned reply
eval/
  label_schema.md         how the golden set is sampled and labeled
  build_golden_set.py      samples real (unlabeled) candidates for hand-labeling
  build_sample_golden_set.py  builds the offline DEMO golden set (synthetic ground truth)
  sample_golden_set.csv      the demo golden set itself (53 rows)
  metrics.py                intent + escalation metrics
  llm_judge.py               reply-quality rubric judge (LLM / heuristic fallback) + human-agreement calc
  run_eval.py                 orchestrates everything above
scripts/
  build_sample_data.py     generates the synthetic offline dataset
report/report.md            problem framing, results, failure analysis, limitations, next steps
decision_log.md              10-15 non-obvious decisions and why
```

## Known limitations (see report for full failure analysis)

- Demo golden set is synthetic (53 rows) for offline reproducibility, not the
  150-250 hand-labeled real examples the real submission requires.
- Judge-vs-human agreement in this repo's default run is a demo (second
  heuristic scorer standing in for a human) — flagged loudly in the output
  and in the report. Real submission needs actual human ratings.
- Multi-turn threads are collapsed to first-message -> first-reply (decision_log.md #3).
