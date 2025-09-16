import os
import google.generativeai as genai

SYSTEM = (
    "You are an email triage classifier. "
    "Given an email, decide two things: importance and reply_needed. "
    "Provide scores in [0,1] as floats and short rationale. "
    "Importance means how urgent or impactful this is to the user. "
    "Reply_needed means whether the user should send a response."
)

def get_model():
    api_key = os.getenv("GOOGLE_GENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GENAI_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def classify(email_text: str) -> dict:
    model = get_model()
    prompt = f"""{SYSTEM}

Email:
"""{email_text}"""

Return JSON only with keys: importance, reply_needed, importance_score, reply_needed_score, rationale.
"""
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    # Try to find JSON block
    import json, re
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0) if m else text)
    except Exception:
        data = {
            "importance": False,
            "reply_needed": False,
            "importance_score": 0.0,
            "reply_needed_score": 0.0,
            "rationale": text[:500]
        }
    # Coerce fields
    data["importance"] = bool(data.get("importance", data.get("importance_score", 0) > 0.6))
    data["reply_needed"] = bool(data.get("reply_needed", data.get("reply_needed_score", 0) > 0.6))
    data["importance_score"] = float(data.get("importance_score", 0.0))
    data["reply_needed_score"] = float(data.get("reply_needed_score", 0.0))
    return data

def answer_question(context_text: str, question: str) -> str:
    model = get_model()
    prompt = f"""You are a helpful inbox analyst. Use only the context provided to answer the question.

Context:
"""{context_text}"""

Question: {question}

If you are not sure, say you are not sure from this context.
"""
    resp = model.generate_content(prompt)
    return resp.text.strip()
