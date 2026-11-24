
import time
import uuid

from copy import deepcopy


class ConversationRunner:

    def __init__(self, response_function):

        if not callable(response_function):

            raise TypeError(
                "response_function must be callable"
            )

        self.response_function = response_function


    def run(self, user_turns):

        if not isinstance(user_turns, list):

            raise TypeError(
                "user_turns must be a list"
            )

        if not user_turns:

            raise ValueError(
                "Conversation cannot be empty"
            )

        session_id = str(uuid.uuid4())

        history = []

        turn_results = []

        session_start = time.perf_counter()

        for turn_number, user_text in enumerate(
            user_turns,
            start=1
        ):

            if (
                not isinstance(user_text, str)
                or not user_text.strip()
            ):

                raise ValueError(
                    "Invalid user message"
                )

            history.append({

                "role": "user",

                "content": user_text
            })

            start = time.perf_counter()

            response = self.response_function(
                deepcopy(history)
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            if not isinstance(response, str):

                raise TypeError(
                    "Model response must be a string"
                )

            history.append({

                "role": "assistant",

                "content": response
            })

            turn_results.append({

                "turn_number": turn_number,

                "user_input": user_text,

                "assistant_response": response,

                "latency_ms": round(
                    latency_ms,
                    3
                )
            })

        total_latency = (
            time.perf_counter() - session_start
        ) * 1000

        return {

            "session_id": session_id,

            "conversation_turns": len(
                user_turns
            ),

            "conversation_history": history,

            "turn_results": turn_results,

            "final_response": (
                turn_results[-1]
                ["assistant_response"]
            ),

            "total_latency_ms": round(
                total_latency,
                3
            ),

            "execution_status": "completed"
        }


def mock_response_function(history):

    current_message = history[-1]["content"]

    if "library" in current_message.lower():

        return (
            "The library provides borrowing "
            "and study facilities."
        )

    return (
        "Please consult the relevant "
        "institutional policy."
    )


if __name__ == "__main__":

    runner = ConversationRunner(
        mock_response_function
    )

    result = runner.run([

        "What facilities does the library provide?",

        "Where can I find institutional policies?",

        "Thank you for the information."
    ])

    for turn in result["turn_results"]:

        print(
            "\nTurn:",
            turn["turn_number"]
        )

        print(
            "User:",
            turn["user_input"]
        )

        print(
            "Assistant:",
            turn["assistant_response"]
        )

        print(
            "Latency:",
            turn["latency_ms"],
            "ms"
        )

    print(
        "\nTotal turns:",
        result["conversation_turns"]
    )

    print(
        "Execution status:",
        result["execution_status"]
    )
