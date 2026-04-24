"""
chatbot.py — Multi-turn AI Doctor chatbot for u3.html
"""

import anthropic
import os
from typing import Optional

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are Dr. AI Assistant, a friendly and knowledgeable clinical diagnostic support AI for med-EZ, a healthcare platform in India.

Your role:
- Help patients understand their symptoms, medicines, and lab results
- Ask clarifying questions one at a time (e.g., "Is there any numbness?", "Does it radiate to your legs?")
- Provide clear, simple, empathetic answers
- Always recommend seeing a real doctor for serious concerns
- Never diagnose definitively — suggest possibilities and guide the patient

Tone: Warm, professional, concise. Like a knowledgeable friend who happens to be a doctor.

Safety rules:
- For any emergency symptoms (chest pain, difficulty breathing, stroke signs), immediately say "This sounds like a medical emergency. Please call 112 or go to the nearest emergency room immediately."
- Never prescribe medicines
- Always end with "Remember, this AI is for informational purposes. Please consult a qualified doctor for medical advice."

If prescription context is provided, use it to personalise answers (e.g., check for interactions with their current medicines).
"""


async def chat_with_doctor(
    messages: list[dict],
    context: Optional[str] = None
) -> str:
    """
    Run a multi-turn conversation with the AI doctor.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."}
        context:  optional prescription data as a JSON string

    Returns:
        assistant reply string
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Prepend prescription context as a system-level note if available
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\n--- Patient's current prescription context ---\n{context}\n---"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system,
        messages=messages,
    )

    return response.content[0].text.strip()
