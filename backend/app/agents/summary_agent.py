from app.core.llm import client, GROQ_MODEL
from app.utils.json_parser import parse_llm_json


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "string",
        },
        "objectives": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "key_highlights": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "executive_summary",
        "objectives",
        "key_highlights",
    ],
    "additionalProperties": False,
}


def _call_llm(prompt: str, max_tokens: int) -> str:
    """
    Call Groq using structured JSON output.

    GPT-OSS is a reasoning model, so:
      - reasoning_effort is kept low
      - include_reasoning=False keeps the actual answer in content
      - JSON Schema guarantees the returned structure
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "document_summary",
                "schema": SUMMARY_SCHEMA,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content

    print("[summary_agent] RAW LLM RESPONSE:")
    print(repr(content))

    return content


def summary_agent(state):
    """
    Generate:
      - executive summary
      - objectives
      - key highlights

    The summary only uses the opening portion of the document.
    Full-document extraction happens separately in the combined
    extraction agent.
    """

    text = state["text"][:5000]

    document_type_data = state.get("document_type", {})

    doc_type = document_type_data.get(
        "document_type_label",
        document_type_data.get(
            "document_type",
            "document",
        ),
    )

    prompt = f"""
You are an expert document analyst.

You are analyzing a {doc_type}.

Read the document carefully and extract the following:

1. executive_summary

Write 2-4 sentences covering:
- who issued the document, if stated
- what the document is about
- what the issuer needs
- important financial, timeline, or contract context when explicitly stated

2. objectives

Return up to 6 short bullet points describing the issuer's
stated goals or purpose.

Each objective should be under 15 words.

Only include objectives actually supported by the document.

3. key_highlights

Return up to 8 short bullet points containing important details
that are NOT simply scope, deadlines, staffing, or compliance.

Examples include:
- evaluation criteria
- eligibility signals
- budget signals
- contract structure
- special conditions
- unusual requirements
- important operational constraints

Each highlight should be under 20 words.

GROUNDING RULES:

- Use real names, numbers, dates, percentages, and requirements
  from the document.
- Do not invent anything.
- If something is not stated, do not claim that it is stated.
- Do not make assumptions merely because something is common
  for this type of document.

Return ONLY the requested structured JSON.

DOCUMENT:
{text}
"""

    try:
        content = _call_llm(
            prompt,
            max_tokens=1200,
        )

        parsed = parse_llm_json(content)

        if (
            isinstance(parsed, dict)
            and "executive_summary" in parsed
            and "objectives" in parsed
            and "key_highlights" in parsed
        ):
            state["summary"] = parsed
            return state

        print(
            "[summary_agent] Primary structured response was invalid."
        )

    except Exception as e:
        print(
            f"[summary_agent] Primary call failed: {e}"
        )

    # ------------------------------------------------------------------
    # Retry with a much simpler prompt.
    # ------------------------------------------------------------------

    print(
        "[summary_agent] Retrying with simpler prompt..."
    )

    simple_prompt = f"""
Analyze this {doc_type}.

Return ONLY valid structured JSON with:

- executive_summary: one concise paragraph
- objectives: short list of the main goals
- key_highlights: short list of important details

Use only information actually present in the document.
Do not invent anything.

DOCUMENT:
{text[:3500]}
"""

    try:
        content2 = _call_llm(
            simple_prompt,
            max_tokens=800,
        )

        print(
            "[summary_agent] RAW RETRY RESPONSE:"
        )
        print(repr(content2))

        parsed = parse_llm_json(content2)

        if (
            isinstance(parsed, dict)
            and "executive_summary" in parsed
            and "objectives" in parsed
            and "key_highlights" in parsed
        ):
            state["summary"] = parsed
            return state

    except Exception as e:
        print(
            f"[summary_agent] Retry failed: {e}"
        )

    # ------------------------------------------------------------------
    # Last-resort fallback.
    # ------------------------------------------------------------------

    print(
        "[summary_agent] Both attempts failed — using text fallback"
    )

    state["summary"] = {
        "executive_summary": (
            text[:400].strip() + "..."
            if len(text) > 400
            else text.strip()
        ),
        "objectives": [],
        "key_highlights": [],
    }

    return state