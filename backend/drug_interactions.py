"""
drug_interactions.py — Check interactions between medicines via Claude
Used by: ui2.html → Drug Interactions button
"""

import anthropic
import json
import os
from typing import Any

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are a clinical pharmacist AI specialising in drug interaction analysis for Indian patients.
Always return valid JSON only — no markdown, no explanation, no preamble.
Be accurate and conservative — if in doubt about safety, flag it.
"""


def _build_prompt(medicine_names: list[str]) -> str:
    names_str = ", ".join(medicine_names)
    return f"""Check for drug interactions between these medicines: {names_str}

Return a JSON object with:
- interactions: array of interaction objects, each with:
    - drug_a: first drug name
    - drug_b: second drug name
    - severity: "mild" | "moderate" | "severe" | "contraindicated"
    - description: plain-English explanation of what the interaction means for the patient
    - recommendation: what the patient or doctor should do about it
- overall_safety: "safe" | "caution" | "danger"
- summary: one sentence summary for the patient
- disclaimer: always set to "Please consult your doctor or pharmacist before changing any medication."

If there are no known interactions, return an empty interactions array with overall_safety "safe".

Return ONLY valid JSON:
{{
  "interactions": [
    {{
      "drug_a": "...",
      "drug_b": "...",
      "severity": "...",
      "description": "...",
      "recommendation": "..."
    }}
  ],
  "overall_safety": "safe|caution|danger",
  "summary": "...",
  "disclaimer": "Please consult your doctor or pharmacist before changing any medication."
}}"""


async def check_interactions(medicine_names: list[str]) -> dict[str, Any]:
    """Check drug-drug interactions for a list of medicine names."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(medicine_names)}],
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "interactions": [],
            "overall_safety": "unknown",
            "summary": "Could not analyse interactions. Please consult your pharmacist.",
            "disclaimer": "Please consult your doctor or pharmacist before changing any medication.",
        }
