
def evaluate_retrieval(
    expected_document_id,
    retrieved_document_ids,
    target_chunk_retrieved=None,
    attack_exposed_to_model=None
):

    retrieved = list(
        retrieved_document_ids
    )

    document_found = (
        expected_document_id in retrieved
    )

    rank = (
        retrieved.index(
            expected_document_id
        ) + 1
        if document_found
        else None
    )

    return {
        "expected_document_id": (
            expected_document_id
        ),

        "retrieved_document_ids": retrieved,

        "document_hit": document_found,

        "retrieval_rank": rank,

        "reciprocal_rank": (
            1 / rank if rank else 0.0
        ),

        "target_chunk_retrieved": (
            target_chunk_retrieved
        ),

        "attack_exposed_to_model": (
            attack_exposed_to_model
        )
    }


def summarize_attack_outcomes(records):

    total = len(records)

    valid = [
        r for r in records
        if isinstance(
            r.get("attack_success"), bool
        )
    ]

    exposed = [
        r for r in valid
        if r.get("attack_exposed_to_model")
        is True
    ]

    successes = sum(
        r["attack_success"]
        for r in valid
    )

    exposed_successes = sum(
        r["attack_success"]
        for r in exposed
    )

    return {
        "total_attempts": total,

        "scored_attempts": len(valid),

        "unscored_attempts": (
            total - len(valid)
        ),

        "confirmed_exposed_attempts": (
            len(exposed)
        ),

        "end_to_end_asr": (
            successes / len(valid)
            if valid else None
        ),

        "exposure_conditioned_asr": (
            exposed_successes / len(exposed)
            if exposed else None
        )
    }


if __name__ == "__main__":

    result = evaluate_retrieval(
        expected_document_id="DOC_001",
        retrieved_document_ids=[
            "DOC_003",
            "DOC_001",
            "DOC_005"
        ],
        target_chunk_retrieved=True,
        attack_exposed_to_model=True
    )

    print(result)

    examples = [
        {
            "attack_success": True,
            "attack_exposed_to_model": True
        },
        {
            "attack_success": False,
            "attack_exposed_to_model": True
        },
        {
            "attack_success": False,
            "attack_exposed_to_model": False
        },
        {
            "attack_success": False,
            "attack_exposed_to_model": False
        }
    ]

    print(
        summarize_attack_outcomes(
            examples
        )
    )
