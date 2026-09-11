# Report: NimbusAir Support Agent

Numbers below are from `python -m eval.run_eval --brand NimbusAir --golden eval/sample_golden_set.csv`,
reproducible in one command (see README). Brand is the synthetic `NimbusAir` stand-in
(53 resolved pairs) — see decision_log.md #1 for why, and the real-dataset instructions
in README for swapping in an actual Kaggle brand.

## 1. Problem framing

**What "good" means for this brand, concretely:**
- **Never auto-send a reply that should have gone to a human.** For a support agent, a
  false-negative escalation (silently auto-handling a legal threat, a large refund, or a
  repeat complaint) is much more expensive than an unnecessary human review. I optimized
  the escalation rule engine for recall on the risky cases first, accuracy second.
- **Replies must be traceable to a real historical resolution**, not fluent-sounding
  invented policy. This is why every reply carries `grounded_on` — the tweet_ids of the
  historical examples it was based on — even in template/fallback mode.
- **A wrong intent label is only bad insofar as it produces a wrong reply or a wrong
  escalation call.** I did not chase intent-classification accuracy as an end in itself.

**What I chose not to build:**
- No multi-turn dialogue state / follow-up handling — every message is scored independently.
  Real support is a back-and-forth; this agent only makes the *first* auto-handle/escalate
  call. (decision_log.md #3)
- No sentiment model as a separate signal — sentiment is folded into the escalation keyword
  rules and repeat-contact counting instead of a dedicated classifier, to keep the escalation
  logic auditable as plain rules rather than another opaque score.
- No fine-grained (Banking77-style, 77-label) intent taxonomy — 6 intents, chosen for label
  consistency over granularity (decision_log.md #4).
- No embedding-based retrieval — TF-IDF was sufficient at this brand's data volume
  (decision_log.md #5).

## 2. Results vs. baselines

| | Intent accuracy | Intent macro-F1 | Escalation precision | Escalation recall | Escalation F1 |
|---|---|---|---|---|---|
| **Trivial** (majority class / never escalate) | 0.208 | 0.057 | 0.0 | 0.0 | 0.0 |
| **Simple** (TF-IDF + LogReg, cross-val) | **1.000** | **1.000** | 0.538 | 0.483 | 0.509 |
| **System** (this agent) | 0.868 | 0.829 | **0.765** | 0.448 | 0.565 |

Reply quality (heuristic judge, n=15, no API key in this run): mean overall **4.13/5**
(grounded 5.0, correctness 3.32, tone 3.20, actionable 5.0).

The system clears the trivial baseline by a wide margin on every axis, and clears the simple
baseline on escalation precision/F1. It does **not** beat the simple baseline on intent
accuracy — see section 4, this is the headline number's biggest catch.

## 3. Failure analysis — top 5

1. **Keyword overlap between intents causes systematic confusion.** Top confusion pairs:
   `billing_dispute -> delivery_delay` (3x), `billing_dispute -> general_inquiry` (2x),
   `account_login -> general_inquiry` (2x). Root cause: `"baggage fee"` (billing) shares the
   word "baggage" with the delivery-delay keyword list, and short account/billing questions
   with no complaint verb look identical to general inquiries to a keyword matcher. Confirmed
   independently via clustering (`scripts/derive_intents.py`) — cluster 3 visibly mixes
   baggage-fee billing complaints with pet-travel policy questions on the term "baggage."
   *Hypothesis:* keyword-only classification can't be fixed by adding more keywords — it needs
   either an LLM classifier (already supported, just needs a key) or explicit negative
   keywords ("fee", "charged" as billing-only even when "baggage" is present).

2. **Escalation recall (0.448) is worse than escalation precision (0.765).** The system is
   too conservative on repeat-contact escalation specifically — `REPEAT_CONTACT_ESCALATION_THRESHOLD=3`
   requires literally the 3rd occurrence of near-identical text, which under-fires on real
   traffic where repeat complaints are reworded each time, not copy-pasted.
   *Hypothesis:* repeat-contact detection needs to key off (customer_id, rough topic) within a
   time window, not exact text matching — not implementable well on this synthetic data since
   author_ids are randomly assigned per message here.

3. **The simple TF-IDF+LogReg baseline hits 100% intent accuracy — and that number is
   close to meaningless.** It's cross-validated on the *same 53-row golden set* it was fit on,
   which is small, synthetically templated (only 18 underlying message templates repeated
   with light variation), and has near-zero vocabulary overlap *across* intent classes. A
   linear classifier on TF-IDF features will always look artificially strong under these
   conditions. See section 4.

4. **Template-mode replies (no API key) can be stale or over-specific.** The rule-mode reply
   generator just reuses the single closest historical reply verbatim, including
   brand-specific specifics ("Delays on NA204 were due to weather") that may not apply to the
   new message's actual flight number. This is fine as a grounding-safe fallback but is not
   an acceptable production reply without an LLM rewrite pass — flagged as `reply_method:
   "template"` in every output specifically so it's distinguishable downstream.

5. **Escalation force-keywords are brittle to negation/sarcasm.** `"sue"`, `"fraud"`, etc.
   trigger escalation even in clearly non-threatening usage ("my lawyer friend said this is
   normal lol" would still force-escalate). This is a deliberate false-positive-tolerant
   tradeoff (decision_log.md #10) but it means precision on the *reason given* is lower than
   precision on the escalate/don't-escalate call itself — worth a small adversarial-example
   check before trusting the stated reasons, not just the binary decision.

## 4. What is misleading about my headline number

Several things, stated plainly:

- **The simple baseline's 100% intent accuracy is not a sign it's a better classifier than
  the system.** It's evaluated in-sample (cross-validated on the exact 53 rows it was trained
  on) on synthetic data with only 18 underlying templates and clean vocabulary separation
  between intents. On real, noisier Twitter traffic with typos, slang, and genuinely
  overlapping vocabulary, a 53-example TF-IDF+LogReg model would almost certainly generalize
  far worse than this number suggests. The system's 86.8% is the more honest number here
  specifically because it *wasn't* fit on this data at all (rule-based, or LLM zero-shot).

- **The judge-vs-human agreement number (Pearson r = -0.069) is not real evidence of
  anything**, and I want to be explicit about why it's in the output at all: I don't have an
  independent human rater in this build environment, so `run_eval.py` compares the reply
  judge against a second, independently-coded *heuristic* scorer purely to prove the
  agreement-computation code runs correctly end-to-end. It is not a measurement of judge
  quality. The near-zero correlation is actually the honest result of comparing two
  unrelated heuristics against each other — it says nothing about whether an LLM judge would
  agree with a real human. This must be replaced with real human ratings before any reply-
  quality number here is trusted (decision_log.md #9).

- **The reply-quality mean of 4.13/5 was scored by the heuristic fallback judge**, not an
  LLM, in this run (no API key was configured for this eval). The heuristic judge's
  `grounded` and `actionable` axes are structurally almost-guaranteed to score high (word
  overlap with the retrieved example, presence of "DM"/"please") since the reply generator
  in fallback mode literally reuses historical reply text. This measures "did we copy the
  template correctly," not "is this a good reply." The LLM judge mode is a materially
  different, more skeptical evaluation and would very likely score `correctness` and `tone`
  lower on any reply where the retrieved example doesn't closely match the new message's
  specifics.

- **53 golden examples is well below the 150-250 target**, and all synthetic. Sampled/labeled
  this way because a real hand-labeling session wasn't possible in this build environment
  (no real dataset access) — see README's "Using the real dataset" section for the actual
  process this repo is built to support.

- **Escalation recall/precision numbers depend entirely on how `gold_escalate` was defined**,
  and here it was defined by a rule-based proxy (`eval/build_sample_golden_set.py`), not a
  human support lead's judgment. A real human labeler would likely be *more* conservative
  (escalate more) than this proxy on borderline cases, which would lower the reported recall
  further once real gold labels replace the proxy.

## 5. What I'd do with one more week

1. Run the real pipeline against a real brand's data with an actual `ANTHROPIC_API_KEY`,
   hand-label a real 200-example golden set per `eval/label_schema.md`, and get an actual
   human to independently score ~30 replies for real judge-agreement numbers.
2. Fix the intent-confusion root cause found in failure mode #1 — either move to LLM
   classification by default, or add negative/exclusion keywords derived from the confusion
   matrix itself.
3. Replace exact-text repeat-contact detection with (author_id, time window, topic similarity)
   — doable on the real dataset since real `author_id`s are stable per customer, unlike the
   synthetic data here.
4. Add a light dialogue-state layer so a reply can reference "as I mentioned before" instead
   of treating every message as a cold start — this is the single biggest gap vs. how real
   support conversations actually work (decision_log.md #3).
5. Stress-test the escalation force-keywords against a deliberately adversarial set (sarcasm,
   negation, quoted third-party speech) and tighten precision on the *stated reason* text, not
   just the binary decision, per failure mode #5.
6. Add a small regression suite of the failure-mode examples found here so future changes
   don't silently reintroduce them.
