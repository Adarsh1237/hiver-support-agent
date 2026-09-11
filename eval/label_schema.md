# Golden Eval Set — Label Schema

File: `eval/golden_set.csv` (you build this; `eval/sample_golden_set.csv` is a
small synthetic demo so the harness is runnable out of the box).

| column | meaning |
|---|---|
| `tweet_id` | source tweet id from `data/twcs.csv` |
| `customer_text` | the inbound customer message |
| `gold_intent` | hand-labeled intent, one of `src/config.py:INTENTS` |
| `gold_escalate` | `True`/`False` — should a human handle this, hand-judged |
| `gold_escalate_reason` | free text, 1 short phrase, why |
| `sampled_via` | `stratified` / `random` / `adversarial` (see below) |
| `notes` | anything ambiguous about the label (optional but encouraged) |

## Sampling method (for the real dataset)

Target: 150–250 examples, brand-specific.

1. **Stratified (60%)** — run the rule-based classifier (`src/classifier.py`,
   no API key needed) over all inbound messages for the brand, group by its
   predicted intent, and sample roughly equal counts per intent. This ensures
   the golden set isn't dominated by whichever intent is most common in raw
   traffic (usually "general_inquiry"/complaints skew the raw distribution).
2. **Random (25%)** — pure random sample from the brand's inbound messages,
   unconditioned on the classifier's prediction. This catches intents/phrasing
   the keyword rules under-detect and keeps the stratified sample from
   silently baking in the classifier's own blind spots.
3. **Adversarial (15%)** — hand-picked: sarcasm, multi-intent messages
   ("delayed AND you charged me twice"), non-English or code-switched text,
   very short/ambiguous messages ("??"), and messages that use escalation
   trigger words in a non-triggering way (e.g. "my lawyer friend said this
   is normal, lol"). These are the messages most likely to reveal real
   failure modes — see decision_log.md #7.

## Labeling process

- Labeled by one person (me) in a single sitting to keep labeling policy
  consistent; ambiguous cases get a `notes` entry rather than a coin flip.
- `gold_intent` labeling rule: pick the intent of the customer's PRIMARY ask.
  If a message raises two issues, label the one mentioned first/most
  emphatically, and note the second in `notes` (this is a known lossy
  simplification — flagged in the report's "what's misleading" section).
- `gold_escalate` labeling rule: would I, as a real support lead, be
  comfortable with a bot sending a reply to this customer with zero human
  review? If uncertain, label `True` (escalate) — false negatives (auto-
  handling something that should've escalated) are the worse failure mode.
