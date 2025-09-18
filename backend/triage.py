import json
import ast
import logging
import os
import re
from functools import lru_cache
from typing import Any, Iterable

import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger(__name__)
MODEL_NAME = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")


@lru_cache(maxsize=1)
def _get_owner_context() -> dict[str, Any]:
    name = os.getenv("INBOX_OWNER_NAME", "").strip()
    aliases_env = os.getenv("INBOX_OWNER_ALIASES", "")
    aliases = [alias.strip() for alias in aliases_env.split(",") if alias.strip()]

    display_names: list[str] = []
    if name:
        display_names.append(name)
    display_names.extend(alias for alias in aliases if alias)

    phrase_patterns: list[re.Pattern[str]] = []
    token_terms: set[str] = set()

    for entry in display_names:
        lowered = entry.lower().strip()
        if not lowered:
            continue
        parts = [p for p in re.split(r"[^a-z0-9@]+", lowered) if p]
        if len(parts) > 1:
            phrase_patterns.append(
                re.compile(r"\b" + r"\s+".join(re.escape(part) for part in parts) + r"\b")
            )
        if parts:
            token_terms.update(parts)
        else:
            token_terms.add(lowered)

    token_patterns = [re.compile(rf"\b{re.escape(term)}\b") for term in sorted(token_terms)]

    if display_names:
        if len(display_names) == 1:
            mention_text = display_names[0]
            instruction = (
                " Prioritize situations where the sender directly addresses "
                f"{mention_text} by name as evidence that they expect a response."
            )
            prompt_hint = (
                "The inbox owner goes by {mention_text}. Treat direct mentions of this name as a strong "
                "signal that the sender wants their personal reply."
            ).format(mention_text=mention_text)
        else:
            mention_text = ", ".join(display_names)
            instruction = (
                " Treat direct references to these names as strong signals a personal reply is required: "
                f"{mention_text}."
            )
            prompt_hint = (
                "The inbox owner responds to the following names: {names}. Treat mentions of any of them "
                "as a strong signal the sender expects their personal reply."
            ).format(names=mention_text)
    else:
        instruction = ""
        prompt_hint = ""

    return {
        "name": name,
        "aliases": aliases,
        "display_names": display_names,
        "phrase_patterns": phrase_patterns,
        "token_patterns": token_patterns,
        "instruction": instruction,
        "prompt_hint": prompt_hint,
    }


def _build_classifier_system_instruction() -> str:
    base_instruction = (
        "You are an email triage classifier for a busy professional. "
        "Analyze each email and return structured JSON describing its importance and whether the sender expects the user to respond. "
        "The JSON must contain the keys importance (boolean), importance_score (float 0-1), reply_needed (boolean), reply_needed_score (float 0-1), and rationale (string). "
        "Importance captures urgency or business impact that warrants quick attention. "
        "Reply_needed is true only when the sender clearly awaits a personal response from the user—for example direct questions, requests for confirmation, scheduling coordination, or deliverables. "
        "Treat newsletters, promotions, marketing blasts, receipts, and automated notifications as reply_needed=false unless they explicitly demand that the user reply. "
        "Base your judgement on the sender's intent and context rather than isolated keywords, and explain any uncertainty in the rationale."
    )
    owner_instruction = _get_owner_context()["instruction"]
    if owner_instruction:
        base_instruction += owner_instruction
    return base_instruction


CLASSIFIER_SYSTEM_INSTRUCTION = _build_classifier_system_instruction()

QA_SYSTEM_INSTRUCTION = (
    "You are a helpful inbox analyst. Answer questions using only the provided email context. If the context does not contain the answer, say that you are not sure."
)

ASSISTANT_SYSTEM_INSTRUCTION = (
    "You are Inbox Buddy, a proactive personal email assistant. "
    "When an email almost certainly needs a personal reply, craft a concise notification for the user, highlight the key points they should address, and draft a short, friendly reply they can send. "
    "Always reply in JSON with the keys notification (string under 200 characters addressing the user as 'you'), summary (array of up to three short bullet strings), and reply_draft (string containing a brief email reply written in first person as the user)."
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

ASSISTANT_GENERATION_CONFIG = types.GenerationConfig(
    temperature=0.6,
    top_p=0.9,
    top_k=40,
    max_output_tokens=640,
    response_mime_type="application/json",
)

try:
    SAFETY_FINISH_REASON = types.FinishReason.SAFETY
except AttributeError:  # pragma: no cover - fallback for SDKs without enum
    SAFETY_FINISH_REASON = "SAFETY"


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


@lru_cache(maxsize=1)
def get_assistant_model():
    _configure_client()
    logger.info("Initialized assistant model name=%s", MODEL_NAME)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=ASSISTANT_SYSTEM_INSTRUCTION,
    )


def _refresh_owner_context():
    _get_owner_context.cache_clear()
    global CLASSIFIER_SYSTEM_INSTRUCTION
    CLASSIFIER_SYSTEM_INSTRUCTION = _build_classifier_system_instruction()
    cache_clear = getattr(get_classifier_model, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


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


def _extract_sender_line(email_text: str) -> str:
    for line in email_text.splitlines():
        if line.lower().startswith("from:"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_subject_line(email_text: str) -> str:
    for line in email_text.splitlines():
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip()
    return ""


def _looks_like_marketing(email_text: str) -> bool:
    lowered = email_text.lower()
    marketing_cues = (
        "unsubscribe",
        "view this email in your browser",
        "view in browser",
        "special offer",
        "limited time",
        "sale",
        "deal",
        "% off",
        "discount",
        "coupon",
        "promo code",
        "book now",
        "rent a car",
        "loyalty",
        "rewards",
        "exclusive offer",
        "upgrade now",
        "act now",
        "shop now",
        "learn more",
        "manage your preferences",
        "update your preferences",
        "manage preferences",
        "manage subscription",
        "view this message online",
        "view online",
        "privacy policy",
        "no longer wish to receive",
        "newsletter",
        "digest",
        "webinar",
        "flash sale",
    )
    if any(cue in lowered for cue in marketing_cues):
        return True

    subject = _extract_subject_line(email_text)
    if _is_roundup_subject(subject):
        return True

    sender = _extract_sender_line(email_text).lower()
    sender_cues = (
        "newsletter",
        "no-reply",
        "noreply",
        "updates",
        "offers",
        "promotions",
        "marketing",
        "sales",
        "mailer",
        "notification",
        "@info",
        "@news",
        "@email.",
        "@mail.",
        "@mailer",
        "@campaign",
        "@notify",
        "@marketing",
        "@promo",
        "@deals",
        "@updates.",
        "@messaging",
        "@communication",
    )
    return any(cue in sender for cue in sender_cues)


def _mentions_user_name(email_text: str) -> bool:
    context = _get_owner_context()
    if not context["phrase_patterns"] and not context["token_patterns"]:
        return False

    lowered = email_text.lower()
    for pattern in context["phrase_patterns"]:
        if pattern.search(lowered):
            return True
    for pattern in context["token_patterns"]:
        if pattern.search(lowered):
            return True
    return False


def _is_roundup_subject(subject: str) -> bool:
    if not subject:
        return False
    subject_lower = subject.lower()
    if re.search(r"\b(daily|weekly|monthly|weekend)\s+(digest|newsletter|roundup|summary|update)\b", subject_lower):
        return True
    roundup_cues = (
        "newsletter",
        "digest",
        "roundup",
        "summary",
        "highlights",
        "top stories",
        "top picks",
        "in case you missed",
    )
    if any(cue in subject_lower for cue in roundup_cues):
        return True
    if re.search(r"\b\d{1,3}%\s*(off|discount|savings)\b", subject_lower):
        return True
    if "flash sale" in subject_lower:
        return True
    return False


def _is_no_reply_sender(email_text: str) -> bool:
    lowered = email_text.lower()
    return any(tag in lowered for tag in ("no-reply", "noreply", "do-not-reply", "donotreply"))


def _has_list_unsubscribe(email_text: str) -> bool:
    return "list-unsubscribe" in email_text.lower()


def _has_reply_cue(email_text: str) -> bool:
    if _looks_like_marketing(email_text):
        return False

    lowered = email_text.lower()
    explicit_phrases = (
        "please respond",
        "please reply",
        "please confirm",
        "please let me know",
        "let me know",
        "do let me know",
        "kindly let me know",
        "need your response",
        "awaiting your response",
        "awaiting your reply",
        "awaiting your confirmation",
        "pls advise",
        "please advise",
        "rsvp",
        "let us know",
        "looking forward to your response",
        "looking forward to hearing from you",
        "appreciate your quick response",
        "appreciate your response",
        "any update",
        "any updates",
        "status update",
        "status on",
        "follow up on",
        "following up on",
        "follow up with",
        "follow-up on",
    )
    if any(phrase in lowered for phrase in explicit_phrases):
        return True

    question_lines = [line for line in email_text.splitlines() if "?" in line]
    if not question_lines:
        return False

    marketing_rhetorical = (
        "can you believe",
        "would you like",
        "would you love",
        "can you handle",
        "are you ready",
        "ready for",
        "will you join us",
        "would you join us",
        "can we tempt you",
        "can we interest you",
        "could we interest you",
        "are you excited",
    )
    marketing_keywords = (
        "deal",
        "deals",
        "sale",
        "discount",
        "offer",
        "offers",
        "promo",
        "promotion",
        "upgrade",
        "flash sale",
    )

    pronoun_patterns = (
        r"\byou\b",
        r"\byour\b",
        r"\bwe\b",
        r"\bi\b",
    )
    followup_keywords = (
        "available",
        "availability",
        "able",
        "schedule",
        "calendar",
        "time",
        "date",
        "deadline",
        "status",
        "update",
        "updates",
        "follow up",
        "follow-up",
        "next step",
        "next steps",
        "confirm",
        "confirmation",
        "details",
        "question for",
        "feedback",
        "thoughts",
        "join",
        "joining",
        "attend",
        "attendance",
        "free",
        "coming",
        "call",
        "call me",
        "chat",
        "discuss",
        "discussion",
        "meet",
        "meeting",
        "sync",
        "touch base",
        "connect",
        "share",
        "send",
        "send me",
        "provide",
        "deliver",
        "deliverable",
        "document",
        "documents",
        "doc",
        "deck",
        "slide",
        "slides",
        "report",
        "notes",
        "plan",
        "proposal",
        "quote",
        "invoice",
        "contract",
        "signature",
        "sign",
        "approve",
        "approval",
        "review",
        "take a look",
        "look over",
        "input",
        "decision",
        "answer",
        "help",
        "support",
        "issue",
        "problem",
        "reschedule",
        "make it",
    )
    scheduling_phrases = (
        "can we meet",
        "could we meet",
        "shall we meet",
        "should we meet",
        "can we talk",
        "could we talk",
        "should we talk",
        "can we discuss",
        "could we discuss",
        "should we discuss",
        "can we connect",
        "could we connect",
        "should we connect",
    )

    for line in question_lines:
        lowered_line = line.lower().strip()
        if any(phrase in lowered_line for phrase in marketing_rhetorical):
            continue
        if any(phrase in lowered_line for phrase in explicit_phrases):
            return True
        if any(phrase in lowered_line for phrase in scheduling_phrases):
            return True
        if any(keyword in lowered_line for keyword in marketing_keywords) and not any(
            keyword in lowered_line for keyword in ("schedule", "meeting", "contract", "invoice", "quote")
        ):
            continue
        if any(re.search(pattern, lowered_line) for pattern in pronoun_patterns):
            if any(keyword in lowered_line for keyword in followup_keywords):
                return True
            if re.search(r"\b(what|when|where|who|how|which|why)\b", lowered_line):
                return True

    return False


def _iter_candidate_text(candidate: Any) -> Iterable[str]:
    content = getattr(candidate, "content", None)
    if not content:
        return []
    parts = getattr(content, "parts", None) or []
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
        elif isinstance(part, dict):
            value = part.get("text")
            if value:
                texts.append(value)
    return texts


def _response_to_text(response: Any) -> str:
    pieces: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        if getattr(candidate, "finish_reason", None) == SAFETY_FINISH_REASON:
            continue
        texts = list(_iter_candidate_text(candidate))
        if texts:
            pieces.extend(texts)
            break
    if not pieces:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
        if block_reason:
            logger.warning("Model response blocked by safety systems: %s", block_reason)
    return "\n".join(pieces).strip()


def _strip_code_fence(text: str) -> str:
    if not text:
        return ""
    match = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _escape_unescaped_newlines(snippet: str) -> str:
    """Replace bare newlines inside quoted strings with escaped newlines."""

    if not snippet:
        return ""

    result: list[str] = []
    in_string = False
    quote_char = ""
    escape = False
    pending_cr_in_string = False

    for ch in snippet:
        if pending_cr_in_string:
            if ch == "\n" and in_string:
                pending_cr_in_string = False
                continue
            pending_cr_in_string = False

        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            continue

        if in_string:
            if ch == quote_char:
                in_string = False
                quote_char = ""
                result.append(ch)
            elif ch == "\r":
                result.append("\\n")
                pending_cr_in_string = True
            elif ch == "\n":
                result.append("\\n")
            else:
                result.append(ch)
            continue

        if ch in ('"', "'"):
            in_string = True
            quote_char = ch

        if ch == "\r":
            # Allow CRLF line endings without duplicating carriage returns.
            pending_cr_in_string = False
            continue

        result.append(ch)

    return "".join(result)


def _find_json_block(text: str) -> str | None:
    depth = 0
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
    return None


def _safe_load_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response from model")

    cleaned = _strip_code_fence(text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = _escape_unescaped_newlines(cleaned)

    candidates: list[str] = []
    block = _find_json_block(cleaned)
    if block:
        candidates.append(block)
    candidates.append(cleaned)

    for candidate in candidates:
        snippet = _escape_unescaped_newlines(candidate.strip())
        if not snippet:
            continue
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            try:
                normalized = _escape_unescaped_newlines(snippet.replace("'", '"'))
                return json.loads(normalized)
            except json.JSONDecodeError:
                continue

    pythonish = _escape_unescaped_newlines(cleaned)
    pythonish = re.sub(r"\btrue\b", "True", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
    try:
        obj = ast.literal_eval(pythonish)
    except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to coerce model output to JSON") from exc
    if isinstance(obj, dict):
        return obj
    raise ValueError("Model output was not a JSON object")


def _default_classification(email_text: str, rationale: str) -> dict:
    marketing = _looks_like_marketing(email_text) or _has_list_unsubscribe(email_text)
    reply_cue = _has_reply_cue(email_text)
    name_mentioned = _mentions_user_name(email_text)

    importance = reply_cue and not marketing
    reply_needed = importance
    score = 0.75 if reply_cue and not marketing else 0.1

    if name_mentioned and not marketing:
        importance = True
        score = max(score, 0.55 if reply_cue else 0.35)
        if reply_cue:
            reply_needed = True
            score = max(score, 0.75)
    return {
        "importance": importance,
        "reply_needed": reply_needed,
        "importance_score": score,
        "reply_needed_score": score,
        "rationale": rationale,
        "actionable": importance,
    }


def classify(email_text: str) -> dict:
    model = get_classifier_model()
    owner_context = _get_owner_context()
    prompt = (
        "Classify the following email. Provide the JSON object requested in the system instructions.\n"
        "Email content is enclosed between triple backticks.\n"
        "```\n"
        f"{email_text.strip()}\n"
        "```"
    )
    prompt_hint = owner_context.get("prompt_hint", "")
    if prompt_hint:
        prompt = f"{prompt_hint.strip()}\n\n{prompt}"
    logger.debug("Submitting classification prompt (chars=%d)", len(email_text))
    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config=CLASSIFY_GENERATION_CONFIG,
    )
    try:
        text = (response.text or "").strip()
    except ValueError:
        logger.debug("response.text accessor unavailable; attempting manual extraction")
        text = ""
    if not text:
        text = _response_to_text(response)

    data: dict[str, Any] | None = None
    rationale = ""

    if not text:
        logger.warning(
            "Model returned no text for classification; using heuristic fallback"
        )
        rationale = "Model response was empty."
    else:
        try:
            data = _safe_load_json(text)
        except ValueError as exc:
            logger.warning(
                "Model JSON parse failed (%s); using heuristic fallback", exc
            )
            rationale = text[:500] or str(exc)
        except Exception:
            logger.exception("Failed to parse model response as JSON")
            rationale = text[:500] or "Model response was empty."

    if data is None:
        data = _default_classification(email_text, rationale)

    importance_score = _clamp_score(data.get("importance_score"))
    reply_needed_score = _clamp_score(data.get("reply_needed_score"))

    importance = _coerce_bool(data.get("importance"))
    reply_needed = _coerce_bool(data.get("reply_needed"))

    if importance:
        importance_score = max(importance_score, 0.6)
    elif importance_score >= 0.6:
        importance = True

    if reply_needed:
        reply_needed_score = max(reply_needed_score, 0.6)
    elif reply_needed_score >= 0.6:
        reply_needed = True

    marketing = _looks_like_marketing(email_text) or _has_list_unsubscribe(email_text)
    name_mentioned = _mentions_user_name(email_text)

    if marketing:
        if reply_needed or importance:
            logger.debug("Marketing cues detected; overriding model flags")
        importance = False
        reply_needed = False
        importance_score = min(importance_score, 0.3)
        reply_needed_score = min(reply_needed_score, 0.3)
    else:
        if reply_needed and _is_no_reply_sender(email_text) and reply_needed_score < 0.95:
            logger.debug("No-reply sender detected; overriding reply_needed flag")
            reply_needed = False
            reply_needed_score = min(reply_needed_score, 0.3)

        if reply_needed and importance_score < reply_needed_score:
            importance = True
            importance_score = max(importance_score, reply_needed_score)

        if name_mentioned and reply_needed_score >= 0.35:
            if not reply_needed:
                logger.debug("Name mention detected; promoting reply_needed flag")
            reply_needed = True
            reply_needed_score = max(reply_needed_score, 0.7)
            if importance_score < reply_needed_score:
                importance = True
                importance_score = max(importance_score, reply_needed_score * 0.9)

    actionable = importance and reply_needed

    data["importance"] = importance
    data["reply_needed"] = reply_needed
    data["importance_score"] = importance_score
    data["reply_needed_score"] = reply_needed_score
    data["actionable"] = actionable
    data["rationale"] = str(data.get("rationale", ""))[:500]

    logger.debug(
        "Classification result importance=%.2f reply_needed=%.2f actionable=%s",
        data["importance_score"],
        data["reply_needed_score"],
        actionable,
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


def craft_assistant_message(payload: dict) -> dict:
    sender = payload.get("sender", "Someone")
    subject = payload.get("subject", "(no subject)")
    body = payload.get("body", "")
    snippet = payload.get("snippet", "")

    email_text = f"From: {sender}\nSubject: {subject}\n\n{body}".strip()
    model = get_assistant_model()
    prompt = (
        "A new email probably needs a personal reply. "
        "Summarize it for the user and draft a short reply they can send. "
        "Respond with JSON matching the schema described in the system instruction.\n"
        "Email content is between triple backticks.\n"
        "```\n"
        f"{email_text}\n"
        "```"
    )

    logger.debug(
        "Generating assistant guidance for sender='%s' subject='%s'", sender, subject
    )

    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config=ASSISTANT_GENERATION_CONFIG,
    )

    try:
        text = (response.text or "").strip()
    except ValueError:  # pragma: no cover - SDK defensive path
        logger.debug("response.text accessor unavailable for assistant output")
        text = ""
    if not text:
        text = _response_to_text(response)

    try:
        data = _safe_load_json(text)
    except Exception:
        logger.exception("Failed to parse assistant guidance JSON")
        fallback_summary = snippet or body[:180]
        summary_list = [fallback_summary.strip()] if fallback_summary.strip() else []
        return {
            "notification": f"You have an actionable email from {sender} about '{subject}'.",
            "summary": summary_list,
            "reply_draft": "",
        }

    notification = str(data.get("notification", "")).strip()
    if not notification:
        notification = f"You have an actionable email from {sender}."
    notification = notification[:280]

    summary = data.get("summary", [])
    if isinstance(summary, str):
        summary_items = [
            item.strip(" -*•\t")
            for item in summary.splitlines()
            if item.strip()
        ]
    else:
        summary_items = [str(item).strip() for item in summary if str(item).strip()]
    summary_items = summary_items[:3]

    reply_draft = str(data.get("reply_draft", "")).strip()

    return {
        "notification": notification,
        "summary": summary_items,
        "reply_draft": reply_draft,
    }
