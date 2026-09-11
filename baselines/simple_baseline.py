"""
Simple baseline: TF-IDF + Logistic Regression intent classifier, trained on
the golden set itself (k-fold, since the golden set is small). Reply is a
per-predicted-intent canned template (the single most common historical
agent reply for that intent) -- not grounded per-message, just per-intent.
Never escalates on its own signal; uses the same rule-based escalation
function as the real system so the escalation comparison isn't skewed by
having no escalation logic at all in the "simple" tier.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.escalation import decide as rule_escalate


@dataclass
class SimplePrediction:
    intent: str
    reply: str
    escalate: bool
    escalate_reason: str


class SimpleBaseline:
    def __init__(self, golden_rows, resolved_pairs):
        texts = [r["customer_text"] for r in golden_rows]
        labels = [r["gold_intent"] for r in golden_rows]

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

        # cross-validated predictions on the golden set itself, for honest
        # in-sample metrics reporting (see eval/run_eval.py)
        n_splits = min(5, min(Counter(labels).values()) if labels else 1)
        self.cv_predictions = (
            cross_val_predict(self.model, X, labels, cv=n_splits) if n_splits >= 2 else labels
        )

        # per-intent canned reply = the modal historical agent reply for that intent
        by_intent = defaultdict(Counter)
        for pair in resolved_pairs:
            if pair.gold_intent:
                by_intent[pair.gold_intent][pair.agent_text] += 1
        self.canned_by_intent = {
            intent: counter.most_common(1)[0][0] for intent, counter in by_intent.items()
        }

    def predict(self, message: str) -> SimplePrediction:
        X = self.vectorizer.transform([message])
        intent = self.model.predict(X)[0]
        confidence = float(max(self.model.predict_proba(X)[0]))
        reply = self.canned_by_intent.get(intent, "Thanks for reaching out -- we're looking into this.")
        esc = rule_escalate(message, intent, confidence)
        return SimplePrediction(intent=intent, reply=reply, escalate=esc.escalate, escalate_reason=esc.reason)
