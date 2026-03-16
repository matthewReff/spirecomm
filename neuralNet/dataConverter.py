from spirecomm.spire.card import Card
from spirecomm.spire.potion import Potion
from spirecomm.spire.character import Monster, Player, Orb
from spirecomm.spire.power import Power
from spirecomm.spire.relic import Relic
from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from tensordict import TensorDict, NonTensorStack


def serialize_cards(cards: list[Card]) -> list[TensorDict]:
    serialized_cards = []
    for i, card in enumerate(cards):
        serialized_cards.append(
            {
                "name": card.name,
                "cost": card.cost,
                "index": i,
                "upgraded": card.upgrades,
            }
        )
    return serialized_cards


def serialize_potions(potions: list[Potion]) -> list[TensorDict]:
    serialized_potions = []
    for i, potion in enumerate(potions):
        serialized_potions.append(TensorDict({"name": potion.name, "index": i}))
    return serialized_potions


def serialize_buff(power: Power) -> TensorDict:
    return TensorDict({"name": power.power_name, "amount": power.amount})


def serialize_enemy(monster: Monster) -> TensorDict:
    estimated_damage = monster.move_hits * monster.move_adjusted_damage
    return TensorDict(
        {
            "name": monster.name,
            "location_index": monster.monster_index,
            "health": monster.current_hp,
            "block": monster.block,
            "intent": monster.monster_index,
            "expected_damage": estimated_damage,
            "buffs": map(serialize_buff, monster.powers),
        }
    )


def serialize_orb(orb: Orb) -> str:
    return orb.name


def serialize_relic(relic: Relic) -> str:
    return relic.name


# Translate game state to NN readable format
def game_state_to_NN_input(gameState: Game) -> TensorDict:
    rawDict = {}

    rawDict["relics"] = NonTensorStack(map(serialize_relic, gameState.relics))
    rawDict["potions"] = NonTensorStack(serialize_potions(gameState.potions))

    playerData: Player | None = gameState.player
    rawDict["buffs"] = (
        NonTensorStack(map(serialize_buff, playerData.powers))
        if playerData
        else NonTensorStack([])
    )
    rawDict["current_orbs"] = (
        NonTensorStack(map(serialize_orb, playerData.orbs))
        if playerData
        else NonTensorStack([])
    )
    rawDict["energy"] = playerData.energy if playerData else 0
    rawDict["block"] = playerData.block if playerData else 0
    rawDict["current_health"] = gameState.current_hp
    rawDict["max_health"] = gameState.max_hp

    rawDict["hand_cards"] = NonTensorStack(serialize_cards(gameState.hand))
    rawDict["discarded_cards"] = NonTensorStack(serialize_cards(gameState.discard_pile))
    rawDict["exhausted_cards"] = NonTensorStack(serialize_cards(gameState.exhaust_pile))
    rawDict["remaining_deck_cards"] = NonTensorStack(
        serialize_cards(gameState.draw_pile)
    )

    rawDict["enemies"] = NonTensorStack(map(serialize_enemy, gameState.monsters))

    return TensorDict(rawDict)


# Translate NN output format to readable game state
def NN_output_to_action(networkOutput: TensorDict) -> Action:
    type = networkOutput["type"]
    using_index = networkOutput["using_index"]
    target_index = networkOutput["target_index"]

    if type == 0:
        return EndTurnAction()
    # Card
    elif type == 1:
        return PlayCardAction(card_index=using_index, target_index=target_index)
    # Potion
    elif type == 2:
        return PotionAction(potion_index=using_index, target_index=target_index)

    # Todo flag this as an "invalid state" and punish
    return EndTurnAction()
