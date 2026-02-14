from gymnasium.spaces import Dict, Sequence, Discrete, Text, OneOf
import string

"""
Orbs
0 - Empty
1 - Lightning
2 - Frost
3 - Dark
4 - Plasma
"""
OrbSpace = Discrete(5)

CardSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
PotionSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
ArtifactSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
BuffSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
IntentSpace = Dict(
    {
        "attack": Text(min_length=1, max_length=3, charset=string.digits),
        "effect": Sequence(
            Text(min_length=1, max_length=100, charset=string.ascii_letters)
        ),
    }
)
EnemySpace = Dict(
    {
        "name": Text(min_length=1, max_length=100, charset=string.ascii_letters),
        "intent": Sequence(IntentSpace),
        "health": Text(min_length=1, max_length=3, charset=string.digits),
        "buffs": Sequence(BuffSpace),
    }
)

"""
0 - Self (Includes things with no target)
1-N - Enemy
"""
TargetSpace = Discrete(10)
ActionSpace = OneOf(
    Dict(
        {"target": TargetSpace, "card": CardSpace},
        Dict({"potion": PotionSpace, "target": TargetSpace}),
    )
)

StateSpace = Dict(
    {
        # Directly Interactive
        "hand_cards": Sequence(CardSpace),
        "potions": Sequence(PotionSpace),
        # Personal State
        "current_orbs": Sequence(OrbSpace),
        "energy": Text(min_length=1, max_length=2, charset=string.digits),
        "health": Text(min_length=1, max_length=3, charset=string.digits),
        "artifacts": Sequence(ArtifactSpace),
        "buffs": Sequence(BuffSpace),
        # Additional Card
        "discarded_cards": Sequence(CardSpace),
        "exhausted_cards": Sequence(CardSpace),
        "remaining_deck_cards": Sequence(CardSpace),
        # Enemy
        "enemies": Sequence(EnemySpace),
    }
)
