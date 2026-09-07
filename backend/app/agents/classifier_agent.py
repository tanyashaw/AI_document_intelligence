from app.core.llm import client, GROQ_MODEL
from app.utils.json_parser import parse_llm_json


DOCUMENT_TYPES = [
    "RFP",
    "RFQ",
    "RFI",
    "ITB",
    "SOW",
    "Contract",
    "NDA",
    "Proposal",
    "Policy Document",
    "Report",
    "Other",
]


# Strict structured-output schema.
#
# This prevents GPT-OSS from returning explanatory text around the JSON.
CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": DOCUMENT_TYPES,
        },
        "document_type_label": {
            "type": "string",
        },
        "confidence": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": [
        "document_type",
        "document_type_label",
        "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}


def classifier_agent(state):
    """
    Classify the document using the first ~2000 characters.

    The classifier only needs the opening portion because document
    type is usually identifiable from the title, introduction,
    purpose, issuer, and initial instructions.
    """

    text = state["text"][:2000]

    prompt = f"""
You are an expert document classification system.

Identify the single best document type for the document below.

Choose EXACTLY ONE value from:

{", ".join(DOCUMENT_TYPES)}

Classification rules:

- RFP = Request for Proposal
- RFQ = Request for Quotation
- RFI = Request for Information
- ITB = Invitation to Bid
- SOW = Statement of Work
- Contract = legally binding agreement
- NDA = Non-Disclosure Agreement
- Proposal = a submitted/proposed solution or business proposal
- Policy Document = organizational rules, policies, or procedures
- Report = analytical, informational, status, research, or assessment report
- Other = anything that does not clearly fit the categories above

If you choose "Other", explain what kind of document it actually is
in document_type_label.

Do not invent information.

Keep the reasoning to one short sentence.

DOCUMENT:
{text}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=500,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "document_classifier",
                    "schema": CLASSIFIER_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content

        print(
            "[classifier_agent] RAW LLM RESPONSE:"
        )
        print(repr(content))

        parsed = parse_llm_json(content)

        if (
            not isinstance(parsed, dict)
            or "document_type" not in parsed
        ):
            print(
                "[classifier_agent] Structured response was invalid."
            )

            parsed = {
                "document_type": "Other",
                "document_type_label": "Unknown document",
                "confidence": "Low",
                "reasoning": "The model did not return a valid classification.",
            }

        state["document_type"] = parsed

        return state

    except Exception as e:
        print(
            f"[classifier_agent] Classification failed: {e}"
        )

        state["document_type"] = {
            "document_type": "Other",
            "document_type_label": "Unknown document",
            "confidence": "Low",
            "reasoning": "Classification failed.",
        }

        return state