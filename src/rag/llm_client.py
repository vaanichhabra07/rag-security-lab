
import os
import time

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env", override=True)


class LLMClient:

    def __init__(self, provider=None):

        self.provider = (

            provider

            or os.getenv(
                "LLM_PROVIDER", "gemini"
            )

        ).lower()

        self.temperature = float(

            os.getenv(
                "LLM_TEMPERATURE", "0"
            )
        )

        self.max_output_tokens = int(

            os.getenv(
                "LLM_MAX_OUTPUT_TOKENS",
                "1024"
            )
        )

        if self.provider == "gemini":

            api_key = os.getenv(
                "GEMINI_API_KEY"
            )

            self.model = os.getenv(
                "GEMINI_MODEL"
            )

            if not api_key or not self.model:

                raise ValueError(
                    "Gemini API key or model missing"
                )

            self.client = genai.Client(
                api_key=api_key
            )

        elif self.provider == "openai":

            api_key = os.getenv(
                "OPENAI_API_KEY"
            )

            self.model = os.getenv(
                "OPENAI_MODEL"
            )

            base_url = os.getenv(
                "OPENAI_BASE_URL"
            )

            if not api_key or not self.model:

                raise ValueError(
                    "OpenAI API key or model missing"
                )

            arguments = {

                "api_key": api_key,

                "timeout": 60.0,

                "max_retries": 0
            }

            if base_url:

                arguments["base_url"] = (
                    base_url
                )

            self.client = OpenAI(
                **arguments
            )

        else:

            raise ValueError(

                f"Unsupported provider: "
                f"{self.provider}"
            )

    def generate(
        self,
        system_prompt,
        user_prompt
    ):

        start = time.perf_counter()

        if self.provider == "gemini":

            result = self._generate_gemini(

                system_prompt,

                user_prompt
            )

        else:

            result = self._generate_openai(

                system_prompt,

                user_prompt
            )

        result["generation_latency_ms"] = round(

            (
                time.perf_counter()
                - start
            ) * 1000,

            2
        )

        result["provider"] = self.provider

        result["model"] = self.model

        return result

    def _generate_gemini(
        self,
        system_prompt,
        user_prompt
    ):

        response = (
            self.client.models.generate_content(

                model=self.model,

                contents=user_prompt,

                config=types.GenerateContentConfig(

                    system_instruction=system_prompt,

                    temperature=self.temperature,

                    max_output_tokens=(
                        self.max_output_tokens
                    )
                )
            )
        )

        usage = response.usage_metadata

        answer = response.text or ""

        finish_reason = None

        if response.candidates:

            finish_reason = str(

                response.candidates[0]
                .finish_reason
            )

        return {

            "answer": answer,

            "input_tokens": getattr(

                usage,
                "prompt_token_count",
                None
            ),

            "output_tokens": getattr(

                usage,
                "candidates_token_count",
                None
            ),

            "total_tokens": getattr(

                usage,
                "total_token_count",
                None
            ),

            "finish_reason": finish_reason
        }

    def _generate_openai(
        self,
        system_prompt,
        user_prompt
    ):

        response = (
            self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {
                        "role": "system",
                        "content": system_prompt
                    },

                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=self.temperature,

                max_tokens=self.max_output_tokens
            )
        )

        usage = response.usage

        return {

            "answer": (
                response.choices[0]
                .message.content or ""
            ),

            "input_tokens": getattr(

                usage,
                "prompt_tokens",
                None
            ),

            "output_tokens": getattr(

                usage,
                "completion_tokens",
                None
            ),

            "total_tokens": getattr(

                usage,
                "total_tokens",
                None
            ),

            "finish_reason": (
                response.choices[0]
                .finish_reason
            )
        }
