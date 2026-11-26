
import sys
import json

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.llm_client import LLMClient


def main():

    client = LLMClient()

    result = client.generate(

        system_prompt=(
            "You are a helpful assistant."
        ),

        user_prompt=(
            "Explain retrieval augmented "
            "generation in one sentence."
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    assert result["answer"].strip()

    print("\nLLM CONNECTION TEST PASSED")


if __name__ == "__main__":
    main()
