"""
Samples messages from data/twcs.csv for hand-labeling per the strategy in
label_schema.md (stratified / random / adversarial), and writes an
UNLABELED csv with empty gold_* columns ready for a human to fill in.

python -m eval.build_golden_set --brand NimbusAir --n 200 --out eval/golden_set_unlabeled.csv
"""
import argparse
import csv
import random
from collections import defaultdict

from src import config
from src.data_loader import load_brand_pairs
from src.classifier import classify

random.seed(7)


def build(brand: str, n: int, out_path: str):
    pairs = load_brand_pairs(brand)
    if not pairs:
        raise SystemExit(f"No resolved pairs found for brand={brand}. Run scripts/build_sample_data.py first, "
                          f"or check data/twcs.csv / brand handle.")

    n_stratified = int(n * 0.60)
    n_random = int(n * 0.25)
    n_adversarial = n - n_stratified - n_random  # remainder

    by_predicted_intent = defaultdict(list)
    for pair in pairs:
        pred = classify(pair.customer_text)
        by_predicted_intent[pred.intent].append(pair)

    stratified_sample = []
    intents = list(by_predicted_intent.keys()) or config.INTENTS
    per_intent = max(1, n_stratified // max(1, len(intents)))
    for intent in intents:
        bucket = by_predicted_intent[intent]
        stratified_sample.extend(random.sample(bucket, min(per_intent, len(bucket))))

    remaining = [p for p in pairs if p not in stratified_sample]
    random_sample = random.sample(remaining, min(n_random, len(remaining)))

    # "Adversarial" selection here is heuristic (short/long/multi-punctuation
    # messages) as a starting point -- REPLACE with real hand-picked adversarial
    # examples per label_schema.md #3 before submitting. This script gets you
    # a candidate pool, it doesn't replace human judgment.
    remaining2 = [p for p in remaining if p not in random_sample]
    remaining2.sort(key=lambda p: (len(p.customer_text) < 25, p.customer_text.count("?") + p.customer_text.count("!")), reverse=True)
    adversarial_sample = remaining2[:n_adversarial]

    rows = []
    for sample, tag in [(stratified_sample, "stratified"), (random_sample, "random"), (adversarial_sample, "adversarial")]:
        for pair in sample:
            rows.append({
                "tweet_id": pair.tweet_id,
                "customer_text": pair.customer_text,
                "gold_intent": "",
                "gold_escalate": "",
                "gold_escalate_reason": "",
                "sampled_via": tag,
                "notes": "",
                "_suggested_intent_do_not_trust": classify(pair.customer_text).intent,
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidate rows to {out_path} -- now hand-label gold_intent/gold_escalate/gold_escalate_reason.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=config.DEFAULT_BRAND)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="eval/golden_set_unlabeled.csv")
    args = p.parse_args()
    build(args.brand, args.n, args.out)
