
import re
import time
import unicodedata


class OutputValidator:

    VERSION = "1.0"

    SAFE_RESPONSE = (
        "The generated response could not be "
        "released under the application security "
        "policy. Please rephrase your institutional "
        "question."
    )

    def __init__(self, protected_values=None):

        self.protected_values = tuple(
            value for value in (protected_values or [])
            if isinstance(value, str) and value.strip()
        )

        self.credential_patterns = {
            "private_key_material": re.compile(
                r"-----BEGIN (?:RSA |EC |OPENSSH )?"
                r"PRIVATE KEY-----"
            )
        }

    @staticmethod
    def normalize(text):

        text = unicodedata.normalize("NFKC", text)

        text = text.casefold()

        return " ".join(text.split())

    def validate(self, answer, system_prompt=None):

        if not isinstance(answer, str):
            raise TypeError("Answer must be a string")

        start = time.perf_counter()

        violations = []

        normalized_answer = self.normalize(answer)

        # Check explicitly protected synthetic values.
        for value in self.protected_values:

            if self.normalize(value) in normalized_answer:

                violations.append({
                    "rule_id": "D4_PROTECTED_VALUE",
                    "category": "protected_information",
                    "severity": "high"
                })

                break

        # Check exact disclosure of sufficiently long
        # trusted application instructions.
        if system_prompt:

            normalized_prompt = self.normalize(
                system_prompt
            )

            if (
                len(normalized_prompt) >= 80
                and normalized_prompt in normalized_answer
            ):

                violations.append({
                    "rule_id": "D4_SYSTEM_DISCLOSURE",
                    "category": "instruction_disclosure",
                    "severity": "high"
                })

        # Check recognizable private-key material.
        for category, pattern in (
            self.credential_patterns.items()
        ):

            if pattern.search(answer):

                violations.append({
                    "rule_id": "D4_CREDENTIAL_MATERIAL",
                    "category": category,
                    "severity": "high"
                })

        blocked = bool(violations)

        return {
            "defense": "D4",
            "version": self.VERSION,
            "validation_applied": True,
            "validation_status": (
                "blocked" if blocked
                else "empty_response" if not answer.strip()
                else "passed"
            ),
            "valid": bool(answer.strip()) and not blocked,
            "detected": blocked,
            "blocked": blocked,
            "violations": violations,
            "answer": (
                self.SAFE_RESPONSE if blocked
                else answer
            ),
            "validation_latency_ms": round(
                (time.perf_counter() - start) * 1000,
                4
            )
        }
