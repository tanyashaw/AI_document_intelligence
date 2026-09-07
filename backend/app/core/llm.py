import re
import time

from groq import Groq
from groq import RateLimitError

from app.core.config import GROQ_API_KEY, GROQ_MODEL


_groq = Groq(api_key=GROQ_API_KEY)


class _Completions:
    """
    Small wrapper around Groq chat completions.

    The project originally used llama-3.1-8b-instant. The current
    configuration is intended for reasoning models such as:

        openai/gpt-oss-120b

    The wrapper keeps the old `max_tokens` argument used throughout
    the project, but sends it to Groq as `max_completion_tokens`.

    GPT-OSS is also explicitly configured to:
      - use low reasoning effort
      - avoid returning reasoning separately
      - support structured JSON output
    """

    def create(
        self,
        model: str,
        messages: list,
        temperature: float = 0,
        max_tokens: int = None,
        **kwargs,
    ):
        kwargs_clean = dict(kwargs)

        # GPT-OSS reasoning workloads are better expressed with
        # max_completion_tokens.
        if max_tokens is not None:
            kwargs_clean["max_completion_tokens"] = max_tokens

        # Keep reasoning lightweight for this document-extraction app.
        #
        # Individual agents can override this by explicitly passing
        # reasoning_effort in kwargs.
        kwargs_clean.setdefault("reasoning_effort", "low")

        # We want the useful answer in message.content, not a separate
        # reasoning field.
        kwargs_clean.setdefault("include_reasoning", False)

        max_retries = 4

        for attempt in range(max_retries):
            try:
                response = _groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs_clean,
                )

                # Helpful diagnostic logging while we stabilize the
                # production deployment.
                if response.choices:
                    message = response.choices[0].message

                    content = getattr(message, "content", None)
                    reasoning = getattr(message, "reasoning", None)
                    finish_reason = getattr(
                        response.choices[0],
                        "finish_reason",
                        None,
                    )

                    if not content:
                        print(
                            "[llm] WARNING: model returned empty content"
                        )
                        print(
                            f"[llm] finish_reason={finish_reason}"
                        )

                        if reasoning:
                            print(
                                "[llm] reasoning field was populated "
                                "while content was empty"
                            )

                        usage = getattr(response, "usage", None)

                        if usage:
                            print(
                                "[llm] usage="
                                f"prompt_tokens="
                                f"{getattr(usage, 'prompt_tokens', None)}, "
                                f"completion_tokens="
                                f"{getattr(usage, 'completion_tokens', None)}, "
                                f"total_tokens="
                                f"{getattr(usage, 'total_tokens', None)}"
                            )

                return response

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                # Groq usually tells us how long to wait.
                match = re.search(
                    r"try again in ([\d.]+)s",
                    str(e),
                    re.IGNORECASE,
                )

                wait = (
                    float(match.group(1))
                    if match
                    else (2 ** attempt) * 5
                )

                wait = min(wait + 1, 60)

                print(
                    f"[llm] Rate limited "
                    f"(attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait:.1f}s..."
                )

                time.sleep(wait)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self):
        self.chat = _Chat()


client = _Client()