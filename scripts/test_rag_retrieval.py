
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.retriever import Retriever


def main():

    retriever = Retriever()

    questions = [

        "How many books can a student borrow?",

        "What is the loan duration for library books?",

        "How can I apply for a hostel room?",

        "What is the procedure for medical leave?",

        "How can I request a fee receipt correction?"
    ]

    for question in questions:

        result = retriever.retrieve(

            query=question,

            top_k=3
        )

        print("\n" + "=" * 70)

        print("QUESTION:", question)

        print(

            "Retrieval latency:",

            result["retrieval_latency_ms"],

            "ms"
        )

        for chunk in result["retrieved_chunks"]:

            print("\nRank:", chunk["rank"])

            print(
                "Document:",
                chunk["document_id"]
            )

            print("Domain:", chunk["domain"])

            print(
                "Similarity:",
                round(chunk["similarity"], 4)
            )

            print("Source:", chunk["source"])

            print(
                "Context:",
                chunk["text"][:350]
            )

    print("\nRAG RETRIEVAL TEST COMPLETED")


if __name__ == "__main__":
    main()
