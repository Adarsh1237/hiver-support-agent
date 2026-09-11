"""
Automated (non-LLM-judge) metrics: intent classification and escalation
decision quality against a labeled golden set.
"""
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def intent_metrics(gold_labels, pred_labels, labels):
    acc = accuracy_score(gold_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=labels, average=None, zero_division=0
    )
    per_intent = {
        label: {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    macro_f1 = sum(f1) / len(f1) if len(f1) else 0.0
    return {"accuracy": round(acc, 3), "macro_f1": round(macro_f1, 3), "per_intent": per_intent}


def escalation_metrics(gold_escalate, pred_escalate):
    # gold_escalate=True is the "positive" class we care about catching
    # (false negatives here = system silently auto-handles something risky)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_escalate, pred_escalate, labels=[True, False], average=None, zero_division=0
    )
    tp = sum(1 for g, p in zip(gold_escalate, pred_escalate) if g and p)
    fn = sum(1 for g, p in zip(gold_escalate, pred_escalate) if g and not p)
    fp = sum(1 for g, p in zip(gold_escalate, pred_escalate) if not g and p)
    tn = sum(1 for g, p in zip(gold_escalate, pred_escalate) if not g and not p)
    return {
        "escalate_precision": round(precision[0], 3),
        "escalate_recall": round(recall[0], 3),
        "escalate_f1": round(f1[0], 3),
        "confusion": {"true_positive": tp, "false_negative_DANGEROUS": fn, "false_positive": fp, "true_negative": tn},
        "note": "false_negative_DANGEROUS = gold said escalate, system auto-handled. This is the failure mode that actually costs trust; watch it more than accuracy.",
    }


def confusion_pairs(gold_labels, pred_labels, top_n=5):
    counts = defaultdict(int)
    for g, p in zip(gold_labels, pred_labels):
        if g != p:
            counts[(g, p)] += 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
