
import re
import unicodedata


def normalize_text(text):

    if not isinstance(text, str):

        raise TypeError(
            "Input must be a string"
        )

    normalized = unicodedata.normalize(
        "NFKC",
        text
    )

    normalized = normalized.casefold()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    return normalized


def inspect_text(text):

    normalized = normalize_text(text)

    format_character_count = sum(
        unicodedata.category(char) == "Cf"
        for char in text
    )

    non_ascii_count = sum(
        ord(char) > 127
        for char in text
    )

    return {
        "original_text": text,

        "normalized_text": normalized,

        "normalization_changed_text": (
            text != normalized
        ),

        "original_length": len(text),

        "normalized_length": len(normalized),

        "format_character_count": (
            format_character_count
        ),

        "non_ascii_count": non_ascii_count
    }


if __name__ == "__main__":

    examples = [
        "Library Loan Period",

        "LIBRARY LOAN PERIOD",

        "Library    Loan    Period",

        "Ｌｉｂｒａｒｙ",

        "पुस्तकालय",

        "Bibliothèque"
    ]

    for example in examples:

        print(inspect_text(example))
        print()
