"""
Loads twcs.csv (real Kaggle file, or the synthetic fallback from
scripts/build_sample_data.py -- same schema either way) and builds
resolved (customer_message -> brand_reply) pairs for one brand.

We only keep pairs where a customer inbound tweet has a DIRECT brand reply
(inbound tweet's tweet_id appears in a brand tweet's in_response_to_tweet_id).
Multi-turn threads collapse to first-customer-message -> first-brand-reply,
since that's the unit the agent has to react to. Longer threads are a known
simplification -- see decision_log.md #3 and the report's limitations section.
"""
import csv
from dataclasses import dataclass
from typing import List, Optional

from . import config


@dataclass
class ResolvedPair:
    tweet_id: str
    customer_text: str
    agent_text: str
    gold_intent: Optional[str] = None  # only present in the synthetic sample; None for real data


def _clean(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def load_brand_pairs(brand: str = None, path=None) -> List[ResolvedPair]:
    brand = brand or config.DEFAULT_BRAND
    path = path or config.TWCS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/build_sample_data.py` for a "
            f"synthetic sample, or download twcs.csv from Kaggle into data/."
        )

    rows_by_id = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_id[row["tweet_id"]] = row

    pairs = []
    for row in rows_by_id.values():
        is_inbound = row["inbound"].strip().lower() == "true"
        if not is_inbound:
            continue
        if f"@{brand}".lower() not in row["text"].lower():
            continue
        resp_ids = [r for r in row.get("response_tweet_id", "").split(",") if r]
        agent_reply = None
        for rid in resp_ids:
            resp_row = rows_by_id.get(rid)
            if resp_row and resp_row["inbound"].strip().lower() == "false" \
               and brand.lower() in resp_row["author_id"].lower():
                agent_reply = resp_row["text"]
                break
        if agent_reply is None:
            continue  # unresolved / no direct brand reply -- skip for grounding purposes

        pairs.append(ResolvedPair(
            tweet_id=row["tweet_id"],
            customer_text=_clean(row["text"]),
            agent_text=_clean(agent_reply),
            gold_intent=row.get("_gold_intent") or None,
        ))
    return pairs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=config.DEFAULT_BRAND)
    args = p.parse_args()
    pairs = load_brand_pairs(args.brand)
    print(f"Loaded {len(pairs)} resolved pairs for brand={args.brand}")
    for pair in pairs[:3]:
        print("-", pair.customer_text, "->", pair.agent_text[:80])
