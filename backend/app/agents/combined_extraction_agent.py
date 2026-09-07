"""
Extracts all requirement categories from one document batch.

The extraction is performed in a single Groq call using GPT-OSS
structured JSON output.

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
# Empty result used when a batch cannot be processed.
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
# Strict schema.
#
# IMPORTANT:
# Keep the schema simple.
#
# We do NOT use enums for category/severity/type because a single
# unexpected classification should never cause Groq to reject an
# otherwise useful extraction.
# ----------------------------------------------------------------------

EXTRACTION_SCHEMA = {
    "type": "object",

    "properties": {

        # --------------------------------------------------------------
        # PROJECT SCOPE
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # DEADLINES
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # STAFFING
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # COMPLIANCE
        # --------------------------------------------------------------

        "compliance_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                    },

                    # Deliberately a string rather than enum.
                    #
                    # This prevents the whole batch from failing if
                    # the model uses "Compliance", "Security",
                    # "Privacy", etc.
                    "category": {
                        "type": "string",
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

        # --------------------------------------------------------------
        # DELIVERABLES
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # TECHNICAL REQUIREMENTS
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # COMMERCIAL REQUIREMENTS
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # RISKS
        # --------------------------------------------------------------

        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {

                    "risk": {
                        "type": "string",
                    },

                    # String instead of enum.
                    "severity": {
                        "type": "string",
                    },

                    # String instead of enum.
                    "type": {
                        "type": "string",
                    },

                    # Explicit / Inferred.
                    "source": {
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

    # All eight top-level fields are required.
    #
    # The model MUST return empty arrays when a category has nothing.
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
    Extract all eight requirement categories from one document batch.

    The function intentionally returns an empty result for a failed
    batch rather than crashing the entire document-processing pipeline.
    """

    prompt = f"""
You are an expert document-intelligence extraction system.

Analyze ONLY the DOCUMENT EXCERPT provided below.

Extract information into exactly these 8 categories:

1. project_scope
2. deadlines
3. staffing_requirements
4. compliance_requirements
5. deliverables
6. technical_requirements
7. commercial_requirements
8. risks

If a category has no relevant information in this excerpt,
return an EMPTY ARRAY for that category.

============================================================
IMPORTANT: DO NOT MIX CATEGORIES
============================================================

PROJECT SCOPE
--------------

Use project_scope ONLY for the actual high-level work or purpose
of the project.

Examples:

GOOD:
- Implement an enterprise document management platform
- Migrate legacy documents
- Configure document workflows
- Provide implementation and support services

DO NOT put individual technical specifications here.

For example, these belong in technical_requirements:

- TLS 1.2
- SAML 2.0
- REST API
- OCR accuracy
- browser compatibility
- search response time
- horizontal scaling

------------------------------------------------------------

DEADLINES
---------

Use deadlines ONLY for dates or explicitly stated time periods.

Examples:

- Proposal submission deadline
- Contract award
- Project kickoff
- Design completion
- Go-live
- Warranty period
- Implementation duration

Do not put general requirements here.

------------------------------------------------------------

STAFFING REQUIREMENTS
---------------------

Use staffing_requirements ONLY when the document specifies:

- roles
- personnel
- headcount
- experience
- qualifications
- allocation
- certifications of personnel

------------------------------------------------------------

COMPLIANCE REQUIREMENTS
-----------------------

Use compliance_requirements for requirements that a vendor,
contractor, bidder, or solution MUST satisfy.

Examples:

- legal requirements
- regulatory requirements
- certifications
- insurance
- eligibility
- security/privacy obligations
- submission requirements
- mandatory declarations
- data protection requirements

The category field is a SHORT LABEL such as:

- Legal
- Regulatory
- Certification
- Insurance
- Financial
- Technical
- Security
- Privacy
- Submission
- Eligibility
- Compliance
- Other

Do NOT worry about forcing the label into a predefined list.

------------------------------------------------------------

DELIVERABLES
------------

Use deliverables ONLY for actual outputs/artifacts that the
vendor must produce or hand over.

Examples:

- requirements validation report
- architecture document
- migration plan
- test report
- training materials
- operations handbook
- final implementation report

IMPORTANT:

A technical capability is NOT automatically a deliverable.

For example:

"System shall support SAML 2.0"

belongs in technical_requirements.

It should NOT be a deliverable.

------------------------------------------------------------

TECHNICAL REQUIREMENTS
----------------------

Use technical_requirements for technical/system capabilities.

Examples:

- SAML 2.0
- OpenID Connect
- REST API
- TLS
- encryption
- OCR
- AI classification
- search performance
- browser support
- backup/restore
- scalability
- audit logging
- integrations
- architecture requirements

------------------------------------------------------------

COMMERCIAL REQUIREMENTS
-----------------------

Use commercial_requirements for money, contract, or business terms.

Examples:

- pricing
- payment schedule
- budget
- contract value
- contract duration
- warranty
- penalties
- service credits
- bonds
- guarantees
- commercial negotiation
- recurring fees
- licensing

IMPORTANT:

Contract duration belongs here when discussing the commercial
contract term.

A project milestone such as "go-live on 3 May 2027" belongs
in deadlines.

------------------------------------------------------------

RISKS
------

Extract risks only when they are:

A. explicitly identified in the document

OR

B. reasonably inferable from a specific statement in the document.

Examples:

- tight implementation schedule
- dependency on third-party systems
- migration complexity
- regulatory exposure
- performance constraints
- strict submission requirements
- financial exposure

For each risk provide:

severity:
- High
- Medium
- Low

type:
- Schedule
- Technical
- Financial
- Compliance
- Resource
- Scope
- Legal
- Other

source:
- Explicit
- Inferred

These are labels, not strict values. Use the closest sensible label.

============================================================
EVIDENCE REQUIREMENT
============================================================

EVERY extracted item MUST contain an evidence field.

Evidence must be a SHORT DIRECT QUOTE from the document excerpt.

Maximum approximately 20 words.

Example:

item:
"Support SSO using SAML 2.0 or OpenID Connect"

evidence:
"Support SSO using SAML 2.0 or OpenID Connect"

Do not invent evidence.

If you cannot find supporting text, DO NOT extract the item.

============================================================
PAGE REFERENCES
============================================================

Use the page reference that appears in the document excerpt.

Examples:

"Page 5"
"PAGE 5"

If a page cannot be determined from the excerpt:

"N/A"

Do not invent page numbers.

============================================================
IMPORTANT: EMPTY CATEGORIES
============================================================

You MUST return ALL eight top-level fields.

If there is nothing relevant for a category, return:

[]

Never omit a top-level field.

For example, if there are no risks:

"risks": []

If there are no commercial requirements:

"commercial_requirements": []

============================================================
NO INVENTION
============================================================

Use only information supported by the document excerpt.

Do not invent:

- dates
- prices
- people
- roles
- requirements
- technical specifications
- risks

============================================================
NO DUPLICATE HEADINGS
============================================================

Do not extract generic section headings as requirements.

Prefer the specific requirement underneath a heading.

============================================================
OUTPUT
============================================================

Return ONLY the structured JSON object.

Do not return:

- markdown
- explanations
- headings
- comments
- code fences
- text before JSON
- text after JSON

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

            # This is intentionally reasonably large because this
            # agent can return many extracted records.
            max_tokens=3000,

            reasoning_effort="low",

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
                "Model response was not a dictionary."
            )

            return dict(_RESULT_SHAPE)

        # --------------------------------------------------------------
        # Normalize the response.
        #
        # Even though Structured Outputs should provide all fields,
        # this protects the rest of the application.
        # --------------------------------------------------------------

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
            f"Batch extraction failed: {e}"
        )

        return dict(_RESULT_SHAPE)