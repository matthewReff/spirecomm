from Mods.spirecomm.spirecomm.spire.card import Card
from Mods.spirecomm.spirecomm.spire.potion import Potion
from Mods.spirecomm.spirecomm.spire.character import Monster, Player, Orb
from Mods.spirecomm.spirecomm.spire.power import Power
from Mods.spirecomm.spirecomm.spire.relic import Relic
from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from tensordict import TensorDict


def serialize_cards(cards: list[Card]):
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


def serialize_potion(potion: Potion):
    return potion.name


def serialize_buff(power: Power):
    return {"name": power.power_name, "amount": power.amount}


def serialize_enemy(monster: Monster):
    estimated_damage = monster.move_hits * monster.move_adjusted_damage
    return {
        "name": monster.name,
        "location_index": monster.monster_index,
        "health": monster.current_hp,
        "block": monster.block,
        "intent": monster.monster_index,
        "expected_damage": estimated_damage,
        "buffs": map(serialize_buff, monster.powers),
    }


def serialize_orb(orb: Orb):
    return orb.name


def serialize_relic(relic: Relic):
    return relic.name


# Translate game state to NN readable format
def game_state_to_NN_input(gameState: Game) -> TensorDict:
    rawDict = {}

    playerData: Player = gameState.player
    rawDict["relics"] = map(serialize_relic, gameState.relics)
    rawDict["potions"] = map(serialize_potion, gameState.potions)

    rawDict["buffs"] = map(serialize_buff, playerData.powers)
    rawDict["current_orbs"] = map(serialize_orb, playerData.orbs)
    rawDict["energy"] = playerData.energy
    rawDict["block"] = playerData.block
    rawDict["current_health"] = gameState.current_hp
    rawDict["max_health"] = gameState.max_hp

    rawDict["hand_cards"] = serialize_cards(gameState.hand)
    rawDict["discarded_cards"] = serialize_cards(gameState.discard_pile)
    rawDict["exhausted_cards"] = serialize_cards(gameState.exhaust_pile)
    rawDict["remaining_deck_cards"] = serialize_cards(gameState.draw_pile)

    rawDict["enemies"] = map(serialize_enemy, gameState.monsters)

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
