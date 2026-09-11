"""
Trivial baseline: predicts the majority intent class for everything, replies
with one canned line regardless of content, never escalates.
This is the floor the real system has to clear.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass
class TrivialPrediction:
    intent: str
    reply: str
    escalate: bool


class TrivialBaseline:
    CANNED_REPLY = "Thanks for reaching out -- we're looking into this and will follow up shortly."

    def __init__(self, golden_rows):
        counts = Counter(r["gold_intent"] for r in golden_rows)
        self.majority_intent = counts.most_common(1)[0][0] if counts else "general_inquiry"

    def predict(self, message: str) -> TrivialPrediction:
        return TrivialPrediction(intent=self.majority_intent, reply=self.CANNED_REPLY, escalate=False)
