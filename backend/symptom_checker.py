"""
symptom_checker.py — Analyse patient symptoms via Claude
Used by: ui2.html → Symptom Checker button / u3.html chatbot
"""

import anthropic
import json
import os
from typing import Any, Optional

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are a clinical triage AI assistant for patients in India.
Analyse symptoms and provide helpful, accurate guidance.
Always return valid JSON only — no markdown, no explanation, no preamble.
Never diagnose definitively. Always recommend seeing a real doctor.
Flag emergencies immediately.
"""


def _build_prompt(symptoms: str, age: Optional[int], gender: Optional[str]) -> str:
    patient_info = ""
    if age:
        patient_info += f"Patient age: {age}\n"
    if gender:
        patient_info += f"Patient gender: {gender}\n"

    return f"""{patient_info}Patient symptoms: {symptoms}

Analyse these symptoms and return a JSON object:
{{
  "is_emergency": true or false,
  "emergency_message": "Call 112 immediately. Go to nearest ER." or null,
  "possible_conditions": [
    {{
      "name": "condition name",
      "likelihood": "likely|possible|unlikely",
      "explanation": "plain English explanation for the patient"
    }}
  ],
  "recommended_actions": ["...", "..."],
  "questions_to_ask_doctor": ["...", "..."],
  "home_care_tips": ["...", "..."],
  "urgency": "emergency|see_doctor_today|see_doctor_soon|monitor_at_home",
  "urgency_reason": "brief reason for this urgency level",
  "disclaimer": "This is for informational purposes only. Please consult a qualified doctor."
}}

Keep possible_conditions to maximum 3. Keep all lists to maximum 3 items.
Use plain, simple English suitable for a non-medical patient in India."""


async def check_symptoms(
    symptoms: str,
    age: Optional[int] = None,
    gender: Optional[str] = None
) -> dict[str, Any]:
    """Analyse patient symptoms and return triage guidance."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(symptoms, age, gender)}],
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
            "is_emergency": False,
            "emergency_message": None,
            "possible_conditions": [],
            "recommended_actions": ["Please consult a doctor for a proper evaluation."],
            "questions_to_ask_doctor": [],
            "home_care_tips": [],
            "urgency": "see_doctor_soon",
            "urgency_reason": "Unable to analyse symptoms automatically.",
            "disclaimer": "This is for informational purposes only. Please consult a qualified doctor.",
        }
