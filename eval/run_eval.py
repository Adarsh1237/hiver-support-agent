"""
Runs the full evaluation: system vs. trivial baseline vs. simple baseline,
on the golden set, reporting:
  - intent classification accuracy / macro-F1 (system, trivial, simple)
  - escalation precision/recall (system, simple; trivial never escalates)
  - reply quality via LLM-judge (or heuristic fallback) rubric
  - judge-vs-human agreement on a labeled subset

Usage:
    python -m eval.run_eval --brand NimbusAir --golden eval/sample_golden_set.csv
"""
import argparse
import csv
import json

from src import config
from src.data_loader import load_brand_pairs
from src.pipeline import SupportAgent
from baselines.trivial_baseline import TrivialBaseline
from baselines.simple_baseline import SimpleBaseline
from eval import metrics as M
from eval import llm_judge


def load_golden(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gold_escalate"] = str(r["gold_escalate"]).strip().lower() == "true"
    return rows


def run(brand: str, golden_path: str, judge_sample_n: int = 15):
    golden = load_golden(golden_path)
    resolved_pairs = load_brand_pairs(brand)
    labels = sorted(set(r["gold_intent"] for r in golden))

    agent = SupportAgent(brand)
    trivial = TrivialBaseline(golden)
    simple = SimpleBaseline(golden, resolved_pairs)

    system_preds, trivial_preds, simple_preds = [], [], []
    system_escalate, simple_escalate = [], []
    outputs_for_judge = []

    for row in golden:
        msg = row["customer_text"]
        out = agent.run(msg, exclude_tweet_id=row["tweet_id"])
        system_preds.append(out.intent)
        system_escalate.append(out.escalate)
        outputs_for_judge.append((row, out))

        trivial_preds.append(trivial.predict(msg).intent)

        s_pred = simple.predict(msg)
        simple_preds.append(s_pred.intent)
        simple_escalate.append(s_pred.escalate)

    gold_intents = [r["gold_intent"] for r in golden]
    gold_escalate = [r["gold_escalate"] for r in golden]

    report = {
        "brand": brand,
        "n_golden": len(golden),
        "intent_classification": {
            "system": M.intent_metrics(gold_intents, system_preds, labels),
            "trivial_baseline": M.intent_metrics(gold_intents, trivial_preds, labels),
            "simple_baseline_tfidf_logreg": M.intent_metrics(gold_intents, simple_preds, labels),
        },
        "top_confusions_system": M.confusion_pairs(gold_intents, system_preds),
        "escalation_decision": {
            "system": M.escalation_metrics(gold_escalate, system_escalate),
            "simple_baseline": M.escalation_metrics(gold_escalate, simple_escalate),
            "trivial_baseline": M.escalation_metrics(gold_escalate, [False] * len(gold_escalate)),
        },
    }

    # Reply quality via judge, on a subset (judging every reply is expensive
    # if using a real LLM judge; a subset is standard practice -- see decision_log.md #8)
    judge_subset = outputs_for_judge[:judge_sample_n]
    judge_scores = []
    for row, out in judge_subset:
        examples = agent.retriever.top_k(row["customer_text"], k=3, exclude_tweet_id=row["tweet_id"])
        judge_scores.append(llm_judge.score(row["customer_text"], out.reply, examples))

    report["reply_quality"] = {
        "n_judged": len(judge_scores),
        "judge_method": judge_scores[0].method if judge_scores else None,
        "mean_overall": round(sum(s.overall for s in judge_scores) / len(judge_scores), 2) if judge_scores else None,
        "mean_by_axis": {
            axis: round(sum(getattr(s, axis) for s in judge_scores) / len(judge_scores), 2)
            for axis in ["grounded", "correctness", "tone", "actionable"]
        } if judge_scores else None,
    }

    # DEMO judge-vs-human agreement. See decision_log.md #9 and README:
    # these are NOT independently-collected human ratings, they're a second
    # heuristic scorer used to prove the agreement-computation code works.
    # Replace with real human scores (a CSV of tweet_id,human_overall) before
    # treating this number as evidence.
    demo_human_scores = _demo_human_scores(judge_subset)
    report["judge_vs_human_agreement"] = llm_judge.agreement_with_human(judge_scores, demo_human_scores)
    report["judge_vs_human_agreement"]["WARNING"] = (
        "Demo-only. These 'human' scores were NOT collected from an independent human rater in this "
        "environment -- see decision_log.md #9. Replace eval/run_eval.py:_demo_human_scores with real "
        "human ratings before citing this number in the report."
    )

    return report


def _demo_human_scores(judge_subset):
    """
    Placeholder human scorer for harness demonstration only (see warning
    above). Scores 1-5 based on simple, judge-independent heuristics
    (reply length appropriateness + whether it echoes back the exact intent
    keywords) so it's not just a copy of the judge's own logic.
    """
    scores = []
    for row, out in judge_subset:
        length_ok = 20 <= len(out.reply) <= 320
        mentions_relevant = any(w in out.reply.lower() for w in row["customer_text"].lower().split() if len(w) > 4)
        s = 3.0
        s += 1.0 if length_ok else -0.5
        s += 1.0 if mentions_relevant else -0.5
        scores.append(max(1.0, min(5.0, s)))
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=config.DEFAULT_BRAND)
    p.add_argument("--golden", default="eval/sample_golden_set.csv")
    p.add_argument("--judge-n", type=int, default=15)
    args = p.parse_args()

    result = run(args.brand, args.golden, args.judge_n)
    print(json.dumps(result, indent=2))
