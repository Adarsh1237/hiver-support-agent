"""
End-to-end agent: run(message) -> AgentOutput.

Usage:
    python -m src.pipeline --brand NimbusAir --message "my flight was delayed and no one told me"
"""
import argparse
import json
from dataclasses import dataclass, asdict

from . import config
from .data_loader import load_brand_pairs
from .retrieval import Retriever
from .classifier import classify
from .reply_generator import draft_reply
from .escalation import decide


@dataclass
class AgentOutput:
    message: str
    intent: str
    intent_confidence: float
    classification_method: str
    reply: str
    reply_method: str
    grounded_on: list
    escalate: bool
    escalation_reason: str


class SupportAgent:
    def __init__(self, brand: str = None):
        self.brand = brand or config.DEFAULT_BRAND
        self.pairs = load_brand_pairs(self.brand)
        self.retriever = Retriever(self.pairs)

    def run(self, message: str, prior_contact_count: int = 1, exclude_tweet_id: str = None) -> AgentOutput:
        cls = classify(message)
        examples = self.retriever.top_k(message, k=3, exclude_tweet_id=exclude_tweet_id)
        reply = draft_reply(message, examples)
        esc = decide(message, cls.intent, cls.confidence, prior_contact_count)

        return AgentOutput(
            message=message,
            intent=cls.intent,
            intent_confidence=cls.confidence,
            classification_method=cls.method,
            reply=reply.text,
            reply_method=reply.method,
            grounded_on=reply.grounded_on,
            escalate=esc.escalate,
            escalation_reason=esc.reason,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brand", default=config.DEFAULT_BRAND)
    p.add_argument("--message", required=True)
    args = p.parse_args()

    agent = SupportAgent(args.brand)
    out = agent.run(args.message)
    print(json.dumps(asdict(out), indent=2))
