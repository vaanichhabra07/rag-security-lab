
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class DefenseConfiguration:

    mode: str

    rule_based: bool = False

    llm_detection: bool = False

    context_isolation: bool = False

    output_validation: bool = False

    def active_layers(self):

        layers = []

        if self.rule_based:
            layers.append("D1")

        if self.llm_detection:
            layers.append("D2")

        if self.context_isolation:
            layers.append("D3")

        if self.output_validation:
            layers.append("D4")

        return layers


DEFENSE_CONFIGURATIONS = {

    "D0": DefenseConfiguration("D0"),

    "D1": DefenseConfiguration(
        "D1",
        rule_based=True
    ),

    "D2": DefenseConfiguration(
        "D2",
        llm_detection=True
    ),

    "D3": DefenseConfiguration(
        "D3",
        context_isolation=True
    ),

    "D4": DefenseConfiguration(
        "D4",
        output_validation=True
    ),

    "D5": DefenseConfiguration(
        "D5",
        rule_based=True,
        llm_detection=True,
        context_isolation=True,
        output_validation=True
    )
}


# Pairwise configurations for future ablation studies.
for first, second in combinations(
    ["D1", "D2", "D3", "D4"], 2
):

    mode = f"{first}+{second}"

    layers = {first, second}

    DEFENSE_CONFIGURATIONS[mode] = (
        DefenseConfiguration(
            mode=mode,
            rule_based="D1" in layers,
            llm_detection="D2" in layers,
            context_isolation="D3" in layers,
            output_validation="D4" in layers
        )
    )


class CombinedDefense:

    @staticmethod
    def get_configuration(mode):

        mode = mode.strip().upper()

        if mode not in DEFENSE_CONFIGURATIONS:

            raise ValueError(
                f"Unsupported defense mode: {mode}"
            )

        return DEFENSE_CONFIGURATIONS[mode]

    @staticmethod
    def supported_modes():

        return set(DEFENSE_CONFIGURATIONS)
