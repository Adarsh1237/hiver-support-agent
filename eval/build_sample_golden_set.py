"""
Builds eval/sample_golden_set.csv — a small DEMO golden set for the
synthetic NimbusAir data, using the _gold_intent ground truth that
scripts/build_sample_data.py embedded when generating the synthetic rows.

THIS IS NOT WHAT "150-250 hand-labeled examples" MEANS FOR THE REAL SUBMISSION.
On the real Kaggle dataset there is no ground-truth column -- you must run
eval/build_golden_set.py to get a sample, then hand-label it yourself
(see eval/label_schema.md). This script exists ONLY so the eval harness
(eval/run_eval.py) has something real to run against within the 15-minute
reproduction budget, without requiring a human labeling session first.
See decision_log.md #1 and the report's "what's misleading" section.

Escalation ground truth here is rule-derived (contains force-keyword, OR
mentions a $ amount >= threshold, OR is a 3rd+ repeat complaint) -- a
reasonable proxy for a human support-lead judgment on scripted data, but a
proxy nonetheless. Flagged, not hidden.
"""
import csv
import re
from collections import Counter

from src import config
from src.data_loader import load_brand_pairs

OUT = "eval/sample_golden_set.csv"


def gold_escalate(text: str, intent: str, occurrence_index: int):
    text_l = text.lower()
    for kw in config.ESCALATION_FORCE_KEYWORDS:
        if kw in text_l:
            return True, f"high-risk language ('{kw}')"
    for match in re.findall(r"\$?\s?(\d{2,5})", text):
        if match.isdigit() and int(match) >= config.REFUND_AMOUNT_ESCALATION_THRESHOLD:
            return True, "large monetary amount mentioned"
    if occurrence_index >= config.REPEAT_CONTACT_ESCALATION_THRESHOLD:
        return True, "repeat contact on same issue"
    if intent == "general_inquiry":
        return False, "info question, no complaint"
    return False, "routine complaint, no risk signal"


def main():
    pairs = load_brand_pairs(config.DEFAULT_BRAND)
    seen_counts = Counter()
    rows = []
    for pair in pairs:
        # crude "occurrence index" proxy: how many times this exact template's
        # customer text has appeared so far, simulating repeat-contact signal
        key = pair.customer_text[:40]
        seen_counts[key] += 1
        esc, reason = gold_escalate(pair.customer_text, pair.gold_intent or "general_inquiry", seen_counts[key])
        rows.append({
            "tweet_id": pair.tweet_id,
            "customer_text": pair.customer_text,
            "gold_intent": pair.gold_intent or "general_inquiry",
            "gold_escalate": esc,
            "gold_escalate_reason": reason,
            "sampled_via": "synthetic_demo",
            "notes": "",
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} demo golden rows to {OUT}")


if __name__ == "__main__":
    main()
