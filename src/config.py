import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TWCS_PATH = DATA_DIR / "twcs.csv"

# Which brand to build the agent for. Override with --brand on any script.
DEFAULT_BRAND = os.environ.get("HIVER_BRAND", "NimbusAir")

# If set, the LLM-backed classifier/reply-generator/judge are used.
# If NOT set, everything falls back to deterministic rule-based logic so the
# whole pipeline still runs offline, for free, in <15 min. See decision_log.md #2.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
LLM_MODEL = os.environ.get("HIVER_LLM_MODEL", "claude-sonnet-4-6")

INTENTS = [
    "delivery_delay",       # flight delays, lost/late baggage
    "refund_request",       # refunds, cancellations
    "account_login",        # app/account/login issues
    "product_defect",       # broken seat, entertainment, food quality, etc.
    "billing_dispute",      # wrong/duplicate charges, fee disputes
    "general_inquiry",      # policy questions, no active problem
]

# Keywords used by the rule-based baseline classifier AND as a fallback when
# no API key is configured. Derived by eyeballing clustered sample messages
# (see scripts/derive_intents.py). Not exhaustive by design -- see decision_log.md #4.
INTENT_KEYWORDS = {
    "delivery_delay": ["delay", "delayed", "late", "never showed", "lost my bag",
                        "baggage", "bag never", "pushed back", "didn't arrive"],
    "refund_request": ["refund", "cancel my booking", "cancelled flight", "money back",
                        "reimburse", "still no refund"],
    "account_login": ["log in", "login", "can't log", "invalid credentials", "app crash",
                       "crashing", "logs me out", "check in", "check-in"],
    "product_defect": ["broken", "didn't work", "wasn't working", "inedible", "dead the whole",
                        "tray table", "recline", "entertainment screen"],
    "billing_dispute": ["charged twice", "duplicate charge", "don't recognize", "wrong charge",
                         "baggage fee", "overcharged", "charge on my card"],
    "general_inquiry": ["what's the", "do you fly", "is pet travel", "allowance", "policy"],
}

# Escalation policy: keywords/conditions that force human escalation regardless
# of intent confidence. Kept separate from intent classification on purpose --
# see decision_log.md #6 (escalation is a risk decision, not a confidence score).
ESCALATION_FORCE_KEYWORDS = [
    "lawyer", "sue", "legal action", "fraud", "discriminat", "injur", "hurt me",
    "racist", "assault", "unsafe", "threat",
]
REFUND_AMOUNT_ESCALATION_THRESHOLD = 200  # USD-equivalent; above this, always escalate
LOW_CONFIDENCE_THRESHOLD = 0.55
REPEAT_CONTACT_ESCALATION_THRESHOLD = 3   # 3rd+ message on same thread/topic -> escalate
