# Decision Log

Non-obvious decisions made while building this, and why.

1. **Built a synthetic, schema-identical fallback dataset** (`scripts/build_sample_data.py`)
   instead of requiring the real ~3M-row Kaggle CSV to be present. This environment has no
   network access to Kaggle, and a take-home that only "works in theory" isn't a working
   system. Swapping in the real file is a one-line config change (`data/twcs.csv`), and the
   loader/pipeline code is 100% dataset-agnostic — it was written and tested against the real
   schema, not the synthetic content.

2. **Every LLM-backed component (classifier, reply generator, judge) has a deterministic
   rule/heuristic fallback**, auto-selected when `ANTHROPIC_API_KEY` is unset. This means the
   full pipeline and eval harness reproduce in <15 min with zero API cost or key provisioning,
   and any single API failure mid-eval degrades gracefully instead of crashing the run. Tradeoff:
   the default numbers in this repo are the *weaker* rule-based system's numbers, not the LLM
   system's — flagged prominently in the report.

3. **Multi-turn threads collapse to first-customer-message -> first-brand-reply.** Real support
   threads often take 3-5 exchanges to resolve. Modeling that as a single grounded reply is a
   simplification that overstates how "resolved" a real single-turn auto-reply would be — see
   report's "what's misleading" section.

4. **Intents were kept to 6 and defined by eyeballing clustered messages** (`scripts/derive_intents.py`),
   not by importing Banking77's 77 labels or my own top-down guess. 77 labels is far too granular for a
   Twitter support inbox and would make every unlabeled example a judgment call the labeler can't
   resolve consistently. 6 keeps inter-label boundaries mostly unambiguous at the cost of losing
   fine-grained signal (e.g. "wrong charge" and "duplicate charge" are both `billing_dispute`).

5. **Retrieval uses TF-IDF cosine similarity, not embeddings.** For a single-brand subsample
   (hundreds to low-thousands of resolved pairs) TF-IDF is fast, free, deterministic, and
   good enough — the marginal quality gain from embeddings wasn't worth the added API
   dependency for the core reproducibility path. The retriever is a small isolated class
   (`src/retrieval.py`) specifically so it's a one-file swap to an embedding index later.

6. **Escalation is a separate rule engine, not "escalate when classifier confidence is low."**
   A *confident* classification of a message mentioning legal threats, large refund amounts,
   or repeat contact should still escalate. Folding escalation into classifier confidence would
   hide exactly the cases that matter most (a self-assured wrong answer is worse than an
   uncertain one). Confidence is checked last, only after the risk-specific rules pass.

7. **Golden-set sampling is stratified + random + adversarial (60/25/15), not pure random.**
   Pure random sampling from raw traffic mostly reproduces whatever intent is most common and
   under-samples rare-but-important patterns (legal threats, ambiguous messages). The
   adversarial slice is the one most likely to surface real failure modes, which is the actual
   point of building a golden set at all.

8. **The judge scores a subset (default 15) of golden-set replies, not all of them**, when
   running the eval harness (`--judge-n`). At real LLM-judge scale, judging every reply in a
   250-example set on every eval run is unnecessary cost; a fixed random(ish) subset each run
   is standard practice and cheaper to iterate on. The subset can be raised to the full set for
   a final reported number.

9. **Judge-vs-human agreement in this repo's default output is a DEMO, not real evidence.**
   I don't have an independent human labeler available in this build environment. `run_eval.py`
   computes agreement against a second, independently-coded heuristic scorer
   (`_demo_human_scores`) purely to prove the agreement-computation machinery works end-to-end.
   The output carries an explicit `WARNING` field saying so. This is the single most important
   caveat in the whole repo — see report's "what's misleading" section — and it's called out
   here rather than left implicit.

10. **`gold_escalate` labeling defaults to "escalate" when ambiguous** (see `eval/label_schema.md`).
    A false negative (auto-handling something that should've gone to a human) is a worse failure
    than a false positive (an unnecessary human review), so the golden set is deliberately
    labeled with that asymmetry in mind rather than being "neutral."

11. **The simple baseline reuses the system's own `escalation.py` rule engine** rather than having
    no escalation logic at all. Comparing "system escalation" against "no escalation logic"
    would make the system look artificially better than it is; comparing it against a baseline
    that also tries would isolate the actual value-add (which mostly comes from the intent
    classifier feeding better inputs into the same rule engine, not from smarter escalation
    rules per se).

12. **Reply quality is scored on 4 separate axes (grounded / correctness / tone / actionable)
    rather than one overall 1-5**, because a single number hides *which* failure mode is present.
    A reply can be perfectly toned and totally wrong, or correct and robotic — those need
    different fixes and a single blended score would erase the distinction.

13. **The rule-based classifier's confidence score is deliberately capped below what a real
    calibrated classifier would report** (`min(0.9, ...)` in `classifier.py`), so it can never
    be as confident as the LLM mode. This was a judgment call to keep the escalation threshold
    meaningful in rule-mode: an uncalibrated keyword-hit classifier shouldn't be able to claim
    0.99 confidence and sail through the escalation check.

14. **`_gold_intent` is written into the synthetic CSV but the data loader ignores it entirely**
    for real-dataset parity — the loader code that will run against the real Kaggle file never
    reads or depends on that column, so there's no risk of a code path that silently only works
    on synthetic data.
