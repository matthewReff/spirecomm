from gymnasium.spaces import Dict, Sequence, Discrete, Text, OneOf
import string

"""
0-N Index of action to take
"""
LocationIndexSpace = Discrete(10)

"""
0 - End Turn
1 - Card
2 - Potion
"""
TypeSpace = Discrete(3)

"""
0-N Index of action to take
"""
ActionIndexSpace = Discrete(10)

ActionSpaceKeys = [ "target_index", "using_index", "type" ]
ActionSpace = Dict(
    {
        "target_index": LocationIndexSpace,
        "type": TypeSpace,
        "using_index": ActionIndexSpace
    }
)

"""
Orbs
0 - Empty
1 - Lightning
2 - Frost
3 - Dark
4 - Plasma
"""
OrbSpace = Discrete(5)

CardSpace = Dict(
    {
        "name": Text(min_length=1, max_length=100, charset=string.ascii_letters),
        "cost": Text(min_length=1, max_length=2, charset=string.digits),
        "index": ActionIndexSpace,
        "upgraded": Text(min_length=1, max_length=2, charset=string.digits),
    }
)
PotionSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
RelicSpace = Text(min_length=1, max_length=100, charset=string.ascii_letters)
BuffSpace = Dict(
    {
        "name": Text(min_length=1, max_length=100, charset=string.ascii_letters),
        "amount": Text(min_length=1, max_length=2, charset=string.digits)
    }
)
    
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
        "location_index": LocationIndexSpace,
        "intent": Sequence(IntentSpace),
        "health": Text(min_length=1, max_length=3, charset=string.digits),
        "block": Text(min_length=1, max_length=3, charset=string.digits),
        "expected_damage": Text(min_length=1, max_length=3, charset=string.digits),
        "buffs": Sequence(BuffSpace),
    }
)

StateSpaceKeys = [ "hand_cards", "potions", "current_orbs", "energy", "max_health", "current_health", "block", "relics", "buffs", "discarded_cards", "exhausted_cards", "remaining_deck_cards", "enemies"]
StateSpace = Dict(
    {
        # Directly Interactive
        "hand_cards": Sequence(CardSpace),
        "potions": Sequence(PotionSpace),
        # Personal State
        "current_orbs": Sequence(OrbSpace),
        "energy": Text(min_length=1, max_length=2, charset=string.digits),
        # Health
        "max_health": Text(min_length=1, max_length=3, charset=string.digits),
        "current_health": Text(min_length=1, max_length=3, charset=string.digits),
        "block": Text(min_length=1, max_length=3, charset=string.digits),
        "relics": Sequence(RelicSpace),
        "buffs": Sequence(BuffSpace),
        # Additional Card
        "discarded_cards": Sequence(CardSpace),
        "exhausted_cards": Sequence(CardSpace),
        "remaining_deck_cards": Sequence(CardSpace),
        # Enemy
        "enemies": Sequence(EnemySpace),
    }
)