"""
explanation.py — Plain-language medicine explanations via Claude
"""

import anthropic
import json
import os
from typing import Any

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are a friendly medical assistant helping patients in India understand their prescriptions.
Explain medicines in simple, clear English that a non-medical person can understand.
Be reassuring, accurate, and practical. Focus on what the patient actually needs to know.
Always return valid JSON only — no markdown, no explanation, no preamble.
"""


def _build_user_prompt(medicines: list[dict]) -> str:
    return f"""Explain each medicine below for a patient in India. Use simple, friendly language.

Medicines:
{json.dumps(medicines, indent=2)}

For each medicine return:
- brand_name: same as input (required — used for matching)
- what_it_is: one clear sentence explaining what this medicine does
- how_to_take: plain English instruction combining dose + frequency + instructions
- common_side_effects: array of exactly 2–3 most common side effects in plain language
- important_warning: the single most critical thing to know or avoid, null if nothing critical
- safe_with_food: true if should be taken with food, false if on empty stomach, null if doesn't matter

Return ONLY a JSON array:
[
  {{
    "brand_name": "...",
    "what_it_is": "...",
    "how_to_take": "...",
    "common_side_effects": ["...", "...", "..."],
    "important_warning": "...",
    "safe_with_food": true
  }}
]"""


async def explain_medicines(medicines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate plain-language explanations for extracted medicines."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    if not medicines:
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    slim_meds = [
        {
            "brand_name":   m.get("brand_name", ""),
            "generic_name": m.get("generic_name"),
            "dose":         m.get("dose"),
            "frequency":    m.get("frequency"),
            "duration":     m.get("duration"),
            "route":        m.get("route"),
            "instructions": m.get("instructions"),
        }
        for m in medicines
    ]

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(slim_meds)}],
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "explanations" in result:
            return result["explanations"]
    except json.JSONDecodeError:
        pass

    # Graceful fallback
    return [
        {
            "brand_name":         m.get("brand_name", ""),
            "what_it_is":         None,
            "how_to_take":        f"{m.get('dose', '')} {m.get('frequency', '')}".strip() or None,
            "common_side_effects": [],
            "important_warning":  None,
            "safe_with_food":     None,
        }
        for m in medicines
    ]
