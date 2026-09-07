"""
Extracts ALL 8 requirement categories from one batch of document text
in a SINGLE LLM call.

The extraction uses Groq Structured Outputs with GPT-OSS so that the
model is required to return the exact JSON structure expected by the
rest of the application.

Categories:
    1. project_scope
    2. deadlines
    3. staffing_requirements
    4. compliance_requirements
    5. deliverables
    6. technical_requirements
    7. commercial_requirements
    8. risks
"""

from app.core.llm import client, GROQ_MODEL
from app.utils.json_parser import parse_llm_json


# ----------------------------------------------------------------------
# Default result shape.
# ----------------------------------------------------------------------

_RESULT_SHAPE = {
    "project_scope": [],
    "deadlines": [],
    "staffing_requirements": [],
    "compliance_requirements": [],
    "deliverables": [],
    "technical_requirements": [],
    "commercial_requirements": [],
    "risks": [],
}


# ----------------------------------------------------------------------
# Strict structured-output schema.
#
# Every property is required because Groq Structured Outputs with
# strict=True expects the schema to be explicit.
# ----------------------------------------------------------------------

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "project_scope": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "item",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "deadlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                    },
                    "date": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "event",
                    "date",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "staffing_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                    },
                    "details": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "role",
                    "details",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "compliance_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "Certification",
                            "Legal",
                            "Regulatory",
                            "Insurance",
                            "Financial",
                            "Technical",
                            "Submission",
                            "Eligibility",
                            "Other",
                        ],
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "mandatory": {
                        "type": "boolean",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "requirement",
                    "category",
                    "page_ref",
                    "mandatory",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "deliverables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "item",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "technical_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "item",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "commercial_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "item",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },

        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {
                        "type": "string",
                    },
                    "severity": {
                        "type": "string",
                        "enum": [
                            "High",
                            "Medium",
                            "Low",
                        ],
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "Schedule",
                            "Technical",
                            "Financial",
                            "Compliance",
                            "Resource",
                            "Scope",
                            "Legal",
                            "Other",
                        ],
                    },
                    "source": {
                        "type": "string",
                        "enum": [
                            "Explicit",
                            "Inferred",
                        ],
                    },
                    "page_ref": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
                },
                "required": [
                    "risk",
                    "severity",
                    "type",
                    "source",
                    "page_ref",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
    },

    "required": [
        "project_scope",
        "deadlines",
        "staffing_requirements",
        "compliance_requirements",
        "deliverables",
        "technical_requirements",
        "commercial_requirements",
        "risks",
    ],

    "additionalProperties": False,
}


def extract_all_fields(text: str) -> dict:
    """
    Extract every category from ONE batch of document text.

    Returns a dictionary containing all 8 categories.

    If the model call fails, the batch is skipped instead of crashing
    the entire document analysis.
    """

    prompt = f"""
You are an expert document analyst.

Read the DOCUMENT EXCERPT below and extract every relevant item for
EACH of these 8 categories.

If a category has nothing relevant in this excerpt, return an empty
list for that category.

Do NOT guess or invent information.

============================================================
GROUNDING RULE
============================================================

Every extracted item MUST contain an "evidence" field.

The evidence must be a short direct quote copied from the
DOCUMENT EXCERPT that actually supports the extracted item.

Maximum evidence length: 20 words.

If you cannot find real text supporting an item:

DO NOT INCLUDE THE ITEM.

Do not invent evidence.

============================================================
NO DUPLICATE GRANULARITY
============================================================

If the same requirement appears both as:

1. a generic/summary statement

and

2. specific itemized details,

extract only the specific itemized details.

For example:

Generic:
"Minimum Insurance Requirements"

Specific:
"Commercial General Liability: $2 million per occurrence"

Do NOT create an item for the generic heading if the specific
requirement is already being extracted.

============================================================
1. PROJECT SCOPE
============================================================

Extract the work, goods, services, or activities being requested
or covered.

Fields:

- item
- page_ref
- evidence

============================================================
2. DEADLINES
============================================================

Extract explicitly named:

- submission deadlines
- Q&A deadlines
- milestones
- award dates
- contract start/end dates
- validity periods
- implementation dates
- other explicitly named dates or timeframes

Fields:

- event
- date
- page_ref
- evidence

Rules:

- Only include a milestone if it is explicitly named.
- Do not infer a deadline.
- If the milestone is explicitly named but its date is not given,
  use "Not specified" for date.
- Do not use "N/A" for an unnamed date.

============================================================
3. STAFFING REQUIREMENTS
============================================================

Extract:

- required roles
- required headcount
- qualifications
- certifications
- experience
- key personnel
- team structure

Fields:

- role
- details
- page_ref
- evidence

============================================================
4. COMPLIANCE REQUIREMENTS
============================================================

Extract:

- legal requirements
- regulatory requirements
- certifications
- eligibility
- submission requirements
- insurance
- financial requirements
- required standards

Fields:

- requirement
- category
- page_ref
- mandatory
- evidence

The category MUST be exactly one of:

- Certification
- Legal
- Regulatory
- Insurance
- Financial
- Technical
- Submission
- Eligibility
- Other

Examples:

Certification:
licenses, qualifications, training, professional certifications

Legal:
contract law terms, indemnification, liability clauses

Regulatory:
government or industry regulations/codes

Insurance:
required coverage types and minimum amounts

Financial:
bonding, financial statements, minimum revenue

Technical:
required technical standards/specifications

Submission:
forms, formatting, submission procedure, document sequence

Eligibility:
who is allowed to bid or participate

Other:
anything that genuinely does not fit above

============================================================
5. DELIVERABLES
============================================================

Extract specific outputs, products, reports, services, or artifacts
that must be produced or handed over.

Fields:

- item
- page_ref
- evidence

============================================================
6. TECHNICAL REQUIREMENTS
============================================================

Extract:

- technologies
- systems
- platforms
- technical capabilities
- standards
- methodologies
- integrations
- performance requirements
- security requirements

Fields:

- item
- page_ref
- evidence

============================================================
7. COMMERCIAL REQUIREMENTS
============================================================

Extract:

- pricing
- payment terms
- budget
- contract duration
- warranty
- guarantees
- bonds
- penalties
- commercial financial conditions
- other contract/commercial terms

Fields:

- item
- page_ref
- evidence

============================================================
8. RISKS
============================================================

Extract:

A. Risks explicitly stated in the document.

B. Reasonable risks that can be inferred directly from the
document excerpt.

Examples:

- tight timelines
- complex technical scope
- regulatory exposure
- penalties
- resource constraints
- ambiguous requirements
- large implementation scope

Fields:

- risk
- severity: High | Medium | Low
- type: Schedule | Technical | Financial | Compliance |
        Resource | Scope | Legal | Other
- source: Explicit | Inferred
- page_ref
- evidence

For inferred risks:

- page_ref should be "N/A" unless the page is explicitly known
- evidence must quote the text that caused the inference

Do not invent risks without a reasonable basis in the excerpt.

============================================================
IMPORTANT OUTPUT RULE
============================================================

Return ONLY the structured JSON object.

Do not return:

- markdown
- explanations
- headings
- comments
- ```json fences
- text before the JSON
- text after the JSON

DOCUMENT EXCERPT:

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

            # Larger than the summary/classifier because this response
            # contains up to 8 categories with multiple fields.
            max_tokens=2500,

            # GPT-OSS reasoning is useful for extraction, but we don't
            # need maximum reasoning depth for every batch.
            reasoning_effort="low",

            # Force the response to match EXTRACTION_SCHEMA.
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "document_requirement_extraction",
                    "schema": EXTRACTION_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content

        print(
            "[combined_extraction_agent] RAW LLM RESPONSE:"
        )
        print(repr(content))

        parsed = parse_llm_json(content)

        if not isinstance(parsed, dict):
            print(
                "[combined_extraction_agent] "
                "Structured response was not a dictionary."
            )
            return dict(_RESULT_SHAPE)

        # Make sure every expected category exists.
        result = {}

        for key in _RESULT_SHAPE:
            value = parsed.get(key, [])

            if not isinstance(value, list):
                value = []

            result[key] = value

        return result

    except Exception as e:
        print(
            "[combined_extraction_agent] "
            f"failed on a batch, skipping just that batch: {e}"
        )

        return dict(_RESULT_SHAPE)