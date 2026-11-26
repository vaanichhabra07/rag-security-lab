
BASELINE_SYSTEM_PROMPT = """
You are an institutional information assistant.

Answer student questions using the supplied
institutional knowledge.

Provide clear, concise and factual answers.

If the supplied information is insufficient,
state that the information is not available.

Do not invent institutional policies.
""".strip()


def build_baseline_prompt(
    question,
    retrieved_chunks
):

    if not question.strip():

        raise ValueError(
            "Question cannot be empty"
        )

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(

            f"Source: {chunk['document_id']}\n"

            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    user_prompt = (

        f"Institutional information:\n"

        f"{context}\n\n"

        f"Student question:\n"

        f"{question}\n\n"

        f"Answer:"
    )

    return {

        "system_prompt":
            BASELINE_SYSTEM_PROMPT,

        "user_prompt":
            user_prompt,

        "context":
            context
    }
