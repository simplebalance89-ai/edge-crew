"""
AI Caller for Hurdle System stages.
Uses Azure AI Services endpoint with DeepSeek-V3.2 (fast, cheap)
for controlled/blinded prompts at Stage 3 and Stage 6.
"""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger("edge-crew")

# Azure AI Services config
AZURE_AI_ENDPOINT = os.environ.get(
    "THINKER_ENDPOINT",
    "https://pwgcerp-9302-resource.services.ai.azure.com/openai/v1/"
)
AZURE_AI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")

# DeepSeek-V3.2 for fast structured responses (non-thinking)
HURDLE_MODEL = os.environ.get("HURDLE_AI_MODEL", "DeepSeek-V3.2")
HURDLE_TIMEOUT = 30  # Fast — these are short prompts


async def call_ai(prompt: str, model: str = None) -> str:
    """
    Call AI model with a prompt. Returns response text.
    Uses httpx for async compatibility with the pipeline.
    """
    import httpx

    model = model or HURDLE_MODEL
    endpoint = AZURE_AI_ENDPOINT.rstrip("/")

    # Build request for OpenAI-compatible API
    url = f"{endpoint}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_AI_KEY,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,  # Low temp for consistent scoring
    }

    try:
        async with httpx.AsyncClient(timeout=HURDLE_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.info(f"[AI] {model} responded ({len(content)} chars)")
            return content
    except httpx.TimeoutException:
        logger.warning(f"[AI] {model} timed out after {HURDLE_TIMEOUT}s")
        raise
    except Exception as e:
        logger.warning(f"[AI] {model} failed: {e}")
        raise


async def test_ai():
    """Quick test to verify AI connectivity."""
    try:
        resp = await call_ai("Reply with just the word 'connected' and nothing else.")
        return "connected" in resp.lower()
    except Exception:
        return False
