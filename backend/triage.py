import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger(__name__)
MODEL_NAME = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")

CLASSIFIER_SYSTEM_INSTRUCTION = (
    "You are an email triage classifier for a busy professional. "
    "Analyze each email and return structured JSON describing its importance and whether the sender expects the user to respond. "
    "The JSON must contain the keys importance (boolean), importance_score (float 0-1), reply_needed (boolean), reply_needed_score (float 0-1), and rationale (string). "
    "Importance captures urgency or business impact that warrants quick attention. "
    "Reply_needed is true only when the sender clearly awaits a personal response from the user—for example direct questions, requests for confirmation, scheduling coordination, or deliverables. "
    "Treat newsletters, promotions, marketing blasts, receipts, and automated notifications as reply_needed=false unless they explicitly demand that the user reply. "
    "When the expectation is ambiguous, err on reply_needed=false and explain why in the rationale."
)

QA_SYSTEM_INSTRUCTION = (
    "You are a helpful inbox analyst. Answer questions using only the provided email context. If the context does not contain the answer, say that you are not sure."
)

CLASSIFY_GENERATION_CONFIG = types.GenerationConfig(
    temperature=0.1,
    top_p=0.9,
    top_k=32,
    max_output_tokens=512,
    response_mime_type="application/json",
)

QA_GENERATION_CONFIG = types.GenerationConfig(
    temperature=0.3,
    top_p=0.9,
    top_k=32,
    max_output_tokens=512,
)


@lru_cache(maxsize=1)
def _configure_client() -> str:
    api_key = os.getenv("GOOGLE_GENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GENAI_API_KEY is not set")
    transport = os.getenv("GOOGLE_GENAI_TRANSPORT", "rest").strip().lower() or "rest"
    if transport not in {"rest", "grpc", "grpc_asyncio"}:
        logger.warning("Unsupported transport '%s'; defaulting to REST", transport)
        transport = "rest"
    genai.configure(api_key=api_key, transport=transport)
    logger.info(
        "Configured Google Generative AI client (model=%s transport=%s)", MODEL_NAME, transport
    )
    return transport


@lru_cache(maxsize=1)
def get_classifier_model():
    _configure_client()
    logger.info("Initialized classifier model name=%s", MODEL_NAME)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
    )


@lru_cache(maxsize=1)
def get_qa_model():
    _configure_client()
    logger.info("Initialized QA model name=%s", MODEL_NAME)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=QA_SYSTEM_INSTRUCTION,
    )


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _looks_like_marketing(email_text: str) -> bool:
    lowered = email_text.lower()
    marketing_cues = (
        "unsubscribe",
        "view this email in your browser",
        "view in browser",
    )
    return any(cue in lowered for cue in marketing_cues)


def _is_no_reply_sender(email_text: str) -> bool:
    lowered = email_text.lower()
    return any(tag in lowered for tag in ("no-reply", "noreply", "do-not-reply", "donotreply"))


def classify(email_text: str) -> dict:
    model = get_classifier_model()
    prompt = (
        "Classify the following email. Provide the JSON object requested in the system instructions.\n"
        "Email content is enclosed between triple backticks.\n"
        "```\n"
        f"{email_text.strip()}\n"
        "```"
    )
    logger.debug("Submitting classification prompt (chars=%d)", len(email_text))
    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config=CLASSIFY_GENERATION_CONFIG,
    )
    text = (response.text or "").strip()

    try:
        match = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(match.group(0) if match else text)
    except Exception:
        logger.exception("Failed to parse model response as JSON")
        data = {
            "importance": False,
            "reply_needed": False,
            "importance_score": 0.0,
            "reply_needed_score": 0.0,
            "rationale": text[:500],
        }

    importance_score = _clamp_score(data.get("importance_score"))
    reply_needed_score = _clamp_score(data.get("reply_needed_score"))

    importance = _coerce_bool(data.get("importance")) or importance_score >= 0.6
    reply_needed = _coerce_bool(data.get("reply_needed")) or reply_needed_score >= 0.6

    if importance_score < 0.45:
        importance = False
    if reply_needed_score < 0.45:
        reply_needed = False

    if reply_needed and _looks_like_marketing(email_text):
        logger.debug("Marketing cues detected; overriding reply_needed flag")
        reply_needed = False
        reply_needed_score = min(reply_needed_score, 0.3)

    if reply_needed and _is_no_reply_sender(email_text) and reply_needed_score < 0.95:
        logger.debug("No-reply sender detected; overriding reply_needed flag")
        reply_needed = False
        reply_needed_score = min(reply_needed_score, 0.3)

    data["importance"] = importance
    data["reply_needed"] = reply_needed
    data["importance_score"] = importance_score
    data["reply_needed_score"] = reply_needed_score

    logger.debug(
        "Classification result importance=%.2f reply_needed=%.2f",
        data["importance_score"],
        data["reply_needed_score"],
    )
    return data


def answer_question(context_text: str, question: str) -> str:
    model = get_qa_model()
    prompt = (
        "Context between triple quotes should be used to answer the user's question. "
        "If the answer is not present, reply that you are not sure.\n\n"
        "Context:\n" "\"\"\"\n"
        f"{context_text}\n"
        "\"\"\"\n\n"
        f"Question: {question}"
    )
    logger.debug(
        "Answering question (context_chars=%d, question='%s')",
        len(context_text),
        question,
    )
    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config=QA_GENERATION_CONFIG,
    )
    answer = (response.text or "").strip()
    logger.debug("Answer produced (chars=%d)", len(answer))
    return answer
