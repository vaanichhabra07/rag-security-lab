
import re
import time
import unicodedata


class RuleBasedDefense:

    RULES = [

        {
            "id": "R001",
            "category": "instruction_override",
            "pattern": (
                r"\b(ignore|disregard|forget|override)"
                r"\b.{0,60}\b"
                r"(instructions?|rules?|polic(?:y|ies))\b"
            ),
            "severity": "high"
        },

        {
            "id": "R002",
            "category": "system_information_request",
            "pattern": (
                r"\b(reveal|show|disclose|print)"
                r"\b.{0,50}\b"
                r"(system prompt|hidden instructions|"
                r"secret instructions)\b"
            ),
            "severity": "high"
        },

        {
            "id": "R003",
            "category": "role_reassignment",
            "pattern": (
                r"\b(you are now|act as|pretend to be)"
                r"\b.{0,50}\b"
                r"(administrator|system|developer)\b"
            ),
            "severity": "medium"
        },

        {
            "id": "R004",
            "category": "instruction_priority",
            "pattern": (
                r"\b(highest priority|"
                r"override all instructions|"
                r"new system instruction)\b"
            ),
            "severity": "high"
        },

        {
            "id": "R005",
            "category": "forced_output",
            "pattern": (
                r"\b(output|return|print|respond with)"
                r"\b.{0,25}\b"
                r"(only|exactly|literal text)\b"
            ),
            "severity": "medium"
        }
    ]

    def __init__(self):

        self.compiled_rules = [

            {
                **rule,

                "regex": re.compile(
                    rule["pattern"],
                    re.IGNORECASE | re.DOTALL
                )
            }

            for rule in self.RULES
        ]

    @staticmethod
    def normalize(text):

        text = unicodedata.normalize(
            "NFKC", text
        )

        text = text.casefold()

        text = " ".join(text.split())

        return text

    def scan(
        self,
        text,
        channel="user_input"
    ):

        if not isinstance(text, str):

            raise TypeError(
                "Input must be a string"
            )

        start = time.perf_counter()

        normalized = self.normalize(text)

        matches = []

        for rule in self.compiled_rules:

            match = rule["regex"].search(
                normalized
            )

            if match:

                matches.append({

                    "rule_id": rule["id"],

                    "category": rule["category"],

                    "severity": rule["severity"]
                })

        detected = bool(matches)

        latency = (
            time.perf_counter() - start
        ) * 1000

        return {

            "defense": "D1",

            "channel": channel,

            "detected": detected,

            "matched_rules": matches,

            "match_count": len(matches),

            "action": (
                "block"
                if detected
                else "allow"
            ),

            "detection_latency_ms":
                round(latency, 4)
        }

    def scan_context(self, chunks):

        allowed_chunks = []

        flagged_chunks = []

        scan_results = []

        for chunk in chunks:

            result = self.scan(

                text=chunk["text"],

                channel="retrieved_context"
            )

            scan_results.append({

                "chunk_id": chunk["chunk_id"],

                "document_id": chunk["document_id"],

                **result
            })

            if result["detected"]:

                flagged_chunks.append(chunk)

            else:

                allowed_chunks.append(chunk)

        return {

            "detected": bool(flagged_chunks),

            "allowed_chunks": allowed_chunks,

            "flagged_chunks": flagged_chunks,

            "scan_results": scan_results,

            "original_chunk_count": len(chunks),

            "allowed_chunk_count":
                len(allowed_chunks),

            "flagged_chunk_count":
                len(flagged_chunks)
        }
