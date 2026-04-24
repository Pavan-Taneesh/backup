"""
ocr.py — Prescription image → structured JSON via Claude Vision
"""

import anthropic
import base64
import json
import os
import re
from typing import Any

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are a medical prescription parser with expertise in Indian prescriptions.
Extract structured data from prescription images accurately.
Always return valid JSON only — no markdown, no explanation, no preamble.
If a brand name looks like an Indian drug (e.g. Crocin, Augmentin, Glycomet, Pantop, Dolo),
make your best guess at the generic/active ingredient and set confidence to medium.
"""

USER_PROMPT = """Extract all medicines from this prescription image and return a JSON object.

For each medicine extract:
- brand_name: the name written on the prescription (as-is)
- generic_name: the active ingredient if identifiable, else null
- dose: dosage amount (e.g. "500mg", "10ml"), null if not found
- frequency: how often (e.g. "twice daily", "OD", "BD", "TDS"), null if not found
- duration: how long (e.g. "5 days", "1 month"), null if not found
- route: how taken (e.g. "oral", "topical", "injection"), null if not found
- instructions: special notes (e.g. "after food", "at bedtime"), null if not found

Also extract:
- raw_text: the full readable text visible in the image as a single string
- doctor_name: the prescribing doctor's name if visible, else null
- date: prescription date if visible, else null
- confidence: "high" if clearly readable, "medium" if partially unclear, "low" if mostly unreadable

Return ONLY this JSON structure:
{
  "medicines": [
    {
      "brand_name": "...",
      "generic_name": "...",
      "dose": "...",
      "frequency": "...",
      "duration": "...",
      "route": "...",
      "instructions": "..."
    }
  ],
  "raw_text": "...",
  "doctor_name": "...",
  "date": "...",
  "confidence": "high|medium|low"
}

If the image is not a prescription or is completely unreadable, return:
{ "error": "brief reason why it cannot be parsed" }
"""


def _get_media_type(content_type: str) -> str:
    mapping = {
        "image/jpeg": "image/jpeg",
        "image/png":  "image/png",
        "image/webp": "image/webp",
        "image/gif":  "image/gif",
    }
    return mapping.get(content_type, "image/jpeg")


async def extract_prescription(image_bytes: bytes, content_type: str) -> dict[str, Any]:
    """
    Send prescription image to Claude Vision and return structured extracted data.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if content_type == "application/pdf":
        return {"error": "PDF support coming soon. Please upload a JPG or PNG photo of the prescription."}

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = _get_media_type(content_type)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT}
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": "Could not parse prescription. Please try a clearer image."}
