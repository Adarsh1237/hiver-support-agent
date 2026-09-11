"""
LLM-as-judge for reply quality on a 1-5 rubric across 4 axes:
  - grounded: does the reply match how this brand actually resolves this
    issue (per the retrieved examples), rather than inventing policy?
  - correctness: does it address what the customer actually asked?
  - tone: appropriate for a support reply (empathetic, not robotic/curt)?
  - actionable: does it give the customer a clear next step?

LLM mode: one rubric prompt per reply, key present.
Heuristic fallback (no key): word-overlap with grounding examples (proxy for
"grounded"), length sanity check, presence of an actionable phrase
("DM", "please", a link, a next step verb). This is a WEAKER proxy for
reply quality than the LLM judge -- it cannot catch tone or correctness the
way an LLM can. This gap is exactly the kind of thing to name explicitly in
the report's "what's misleading about my headline number" section if the
eval was run without an API key.
"""
import json
import re
from dataclasses import dataclass, asdict

from src import llm_client

RUBRIC_PROMPT = """You are auditing a customer-support AI agent's draft reply.

Customer message: "{message}"

Historical examples of how this brand has resolved similar issues:
{examples}

Agent's drafted reply: "{reply}"

Score the drafted reply 1-5 on each axis (5 = best):
- grounded: consistent with the brand's historical resolution pattern, no invented policy
- correctness: actually addresses what the customer asked
- tone: empathetic and appropriate, not robotic or dismissive
- actionable: gives the customer a clear next step

Respond ONLY with JSON: {{"grounded": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionable": <1-5>, "rationale": "<one sentence>"}}"""

ACTIONABLE_MARKERS = ["dm", "please", "click", "visit", "contact", "call", "email", "reach out", "let us know"]


@dataclass
class JudgeScore:
    grounded: float
    correctness: float
    tone: float
    actionable: float
    rationale: str
    method: str  # "llm" or "heuristic"

    @property
    def overall(self) -> float:
        return round((self.grounded + self.correctness + self.tone + self.actionable) / 4, 2)


def _heuristic_score(message: str, reply: str, examples) -> JudgeScore:
    reply_words = set(re.findall(r"\w+", reply.lower()))
    if examples:
        example_words = set()
        for ex in examples:
            example_words |= set(re.findall(r"\w+", ex.agent_text.lower()))
        overlap = len(reply_words & example_words) / max(1, len(reply_words))
        grounded = round(1 + 4 * min(1.0, overlap * 2), 2)  # scale overlap into 1-5
    else:
        grounded = 2.0  # no grounding examples available -> can't verify, score low-mid

    msg_words = set(re.findall(r"\w+", message.lower()))
    correctness = round(1 + 4 * min(1.0, len(reply_words & msg_words) / max(1, len(msg_words) * 0.3)), 2)

    tone = 4.0 if any(w in reply.lower() for w in ["sorry", "thanks", "appreciate", "understand"]) else 2.5
    actionable = 5.0 if any(m in reply.lower() for m in ACTIONABLE_MARKERS) else 2.0

    return JudgeScore(
        grounded=grounded, correctness=correctness, tone=tone, actionable=actionable,
        rationale="Heuristic fallback score (no ANTHROPIC_API_KEY set) -- word-overlap and marker-based, "
                   "NOT a substitute for the LLM judge. Treat this run's reply-quality numbers as low-confidence.",
        method="heuristic",
    )


def _llm_score(message: str, reply: str, examples) -> JudgeScore:
    examples_text = "\n".join(f'- Customer: "{e.customer_text}" -> Brand: "{e.agent_text}"' for e in examples) or "(none retrieved)"
    prompt = RUBRIC_PROMPT.format(message=message, examples=examples_text, reply=reply)
    raw = llm_client.complete(prompt, max_tokens=200)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    parsed = json.loads(match.group(0))
    return JudgeScore(
        grounded=float(parsed["grounded"]), correctness=float(parsed["correctness"]),
        tone=float(parsed["tone"]), actionable=float(parsed["actionable"]),
        rationale=parsed.get("rationale", ""), method="llm",
    )


def score(message: str, reply: str, examples) -> JudgeScore:
    if llm_client.is_available():
        try:
            return _llm_score(message, reply, examples)
        except Exception:
            pass
    return _heuristic_score(message, reply, examples)


def agreement_with_human(judge_scores, human_scores):
    """
    Pearson correlation between judge.overall and a parallel list of human
    1-5 overall scores on the SAME replies. Also reports mean absolute
    difference, which is more interpretable than correlation alone on n<30.
    """
    import statistics
    n = len(judge_scores)
    if n < 2:
        return {"error": "need >=2 paired scores"}
    j = [s.overall for s in judge_scores]
    h = list(human_scores)
    mean_j, mean_h = statistics.mean(j), statistics.mean(h)
    cov = sum((j[i] - mean_j) * (h[i] - mean_h) for i in range(n))
    std_j = (sum((x - mean_j) ** 2 for x in j)) ** 0.5
    std_h = (sum((x - mean_h) ** 2 for x in h)) ** 0.5
    pearson = cov / (std_j * std_h) if std_j and std_h else 0.0
    mad = sum(abs(j[i] - h[i]) for i in range(n)) / n
    within_1 = sum(1 for i in range(n) if abs(j[i] - h[i]) <= 1.0) / n
    return {
        "n": n, "pearson_r": round(pearson, 3), "mean_abs_diff": round(mad, 2),
        "pct_within_1_point": round(within_1, 3),
    }
