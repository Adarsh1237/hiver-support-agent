"""
Decides whether a customer message should be auto-handled or escalated
to a human agent.

High-risk situations always escalate. Otherwise, low-confidence
classifications and repeated contacts are escalated.
"""

import re
from dataclasses import dataclass

from . import config


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def _mentions_large_refund_amount(text: str) -> bool:
    for match in re.findall(r"\$?\s?(\d{2,5})", text):
        try:
            if int(match) >= config.REFUND_AMOUNT_ESCALATION_THRESHOLD:
                return True
        except ValueError:
            continue
    return False


def decide(
    message: str,
    intent: str,
    confidence: float,
    prior_contact_count: int = 1,
) -> EscalationDecision:

    text_l = message.lower()

    # 1. High-risk language always goes to a human.
    for kw in config.ESCALATION_FORCE_KEYWORDS:
        if kw in text_l:
            return EscalationDecision(
                True,
                f"Message contains high-risk language ('{kw}') -- routed to human."
            )

    # 2. Large refund/billing amounts require human review.
    if (
        intent in ("refund_request", "billing_dispute")
        and _mentions_large_refund_amount(message)
    ):
        return EscalationDecision(
            True,
            f"Monetary amount is at or above the ${config.REFUND_AMOUNT_ESCALATION_THRESHOLD} threshold."
        )

    # 3. Repeated contacts require human review.
    if prior_contact_count >= config.REPEAT_CONTACT_ESCALATION_THRESHOLD:
        return EscalationDecision(
            True,
            f"This is contact #{prior_contact_count} on the same issue -- repeated contact requires human review."
        )

    # 4. Low-confidence classifications require human review.
    if confidence < config.LOW_CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            True,
            f"Intent confidence ({confidence:.2f}) is below the "
            f"auto-handle threshold ({config.LOW_CONFIDENCE_THRESHOLD})."
        )

    # 5. Normal messages can be auto-handled.
    return EscalationDecision(
        False,
        f"'{intent}' with confidence {confidence:.2f} is above the "
        "auto-handle threshold and no escalation trigger was detected."
    )