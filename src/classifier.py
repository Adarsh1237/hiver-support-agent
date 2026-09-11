"""
Classifies an incoming customer message into one of config.INTENTS.

Two modes, chosen automatically:
  - LLM mode (ANTHROPIC_API_KEY set): zero-shot classification with a fixed
    label set, model self-reports a confidence 0-1.
  - Rule mode (no key): keyword-match scoring over config.INTENT_KEYWORDS.
    Confidence = (matched keywords) / (matched + 1), capped at 0.9, so it's
    never falsely as confident as a "real" classifier -- this matters for
    the escalation logic (see escalation.py). This is the classifier used by
    default so the pipeline is reproducible without any API key.
"""
import json
import re
from dataclasses import dataclass

from . import config
from . import llm_client


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    method: str  # "llm" or "rule"


def _rule_classify(text: str) -> ClassificationResult:
    text_l = text.lower()
    scores = {}
    for intent, keywords in config.INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_l)
        if hits:
            scores[intent] = hits
    if not scores:
        return ClassificationResult(intent="general_inquiry", confidence=0.3, method="rule")
    best_intent = max(scores, key=scores.get)
    hits = scores[best_intent]
    confidence = min(0.9, hits / (hits + 1) + 0.3)
    return ClassificationResult(intent=best_intent, confidence=round(confidence, 2), method="rule")


def _llm_classify(text: str) -> ClassificationResult:
    labels = ", ".join(config.INTENTS)
    prompt = (
        f"Classify this customer support message into exactly one of these intents: {labels}.\n\n"
        f'Message: "{text}"\n\n'
        'Respond ONLY with JSON: {"intent": "<one of the labels>", "confidence": <0.0-1.0>}'
    )
    raw = llm_client.complete(prompt, max_tokens=100)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _rule_classify(text)  # graceful degrade on malformed LLM output
    try:
        parsed = json.loads(match.group(0))
        intent = parsed.get("intent", "general_inquiry")
        if intent not in config.INTENTS:
            intent = "general_inquiry"
        confidence = float(parsed.get("confidence", 0.5))
        return ClassificationResult(intent=intent, confidence=confidence, method="llm")
    except (json.JSONDecodeError, ValueError, TypeError):
        return _rule_classify(text)


def classify(text: str) -> ClassificationResult:
    if llm_client.is_available():
        try:
            return _llm_classify(text)
        except Exception:
            pass  # network/API failure -- degrade to rule-based rather than crash the pipeline
    return _rule_classify(text)
