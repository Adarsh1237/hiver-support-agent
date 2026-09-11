"""
Drafts a reply grounded in how this brand has historically resolved similar
issues (the top-k retrieved resolved pairs from retrieval.Retriever).

LLM mode: few-shots the retrieved (customer_text -> agent_text) pairs and
asks for a new reply in the same voice/policy.
Rule mode (no API key): adapts the single closest-matching historical reply
by swapping the greeting/handle -- crude, but grounded and non-hallucinated,
which is the property we actually need for the fallback path.
"""
from dataclasses import dataclass
from typing import List

from . import llm_client
from .data_loader import ResolvedPair


@dataclass
class DraftedReply:
    text: str
    method: str  # "llm" or "template"
    grounded_on: List[str]  # tweet_ids of the retrieved examples used


def _template_reply(message: str, examples: List[ResolvedPair]) -> DraftedReply:
    if not examples:
        return DraftedReply(
            text="Thanks for reaching out -- we're looking into this and will follow up shortly.",
            method="template",
            grounded_on=[],
        )
    closest = examples[0]
    # crude but grounded: reuse the closest historical resolution's structure
    return DraftedReply(text=closest.agent_text, method="template", grounded_on=[closest.tweet_id])


def _llm_reply(message: str, examples: List[ResolvedPair]) -> DraftedReply:
    shots = "\n\n".join(
        f'Customer: "{ex.customer_text}"\nBrand reply: "{ex.agent_text}"' for ex in examples
    )
    prompt = (
        "You are a customer support agent for this brand. Below are real examples of how "
        "this brand has resolved similar issues in the past. Write a new reply to the current "
        "customer message in the same tone and policy. Keep it under 280 characters, do not "
        "invent policy details not implied by the examples, and ask for booking/account details "
        "via DM if the examples do.\n\n"
        f"Past examples:\n{shots}\n\n"
        f'Current customer message: "{message}"\n\n'
        "Reply:"
    )
    text = llm_client.complete(prompt, max_tokens=200).strip()
    return DraftedReply(text=text, method="llm", grounded_on=[ex.tweet_id for ex in examples])


def draft_reply(message: str, examples: List[ResolvedPair]) -> DraftedReply:
    if llm_client.is_available():
        try:
            return _llm_reply(message, examples)
        except Exception:
            pass
    return _template_reply(message, examples)
