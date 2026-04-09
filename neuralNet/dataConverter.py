from utilities.sqlite_scraping import EncodingMapper
from spirecomm.spire.card import Card
from spirecomm.spire.potion import Potion
from spirecomm.spire.character import Monster, Player, Orb
from spirecomm.spire.power import Power
from spirecomm.spire.relic import Relic
from spirecomm.spire.game import Game
from spirecomm.communication.action import (
    Action,
    EndTurnAction,
    PlayCardAction,
    PotionAction,
)
import logging
import torch


def serialize_cards(
    cards: list[Card], encoding_mapper: EncodingMapper, required_size=10
) -> torch.Tensor:
    serialized_cards = torch.Tensor()
    REQUIRED_ENTRIES = required_size

    for i, card in enumerate(cards):
        this_card = torch.Tensor(
            [encoding_mapper.get_card_encoding(card.name), card.cost, i, card.upgrades]
        )
        serialized_cards = torch.cat((serialized_cards, this_card))

    missing_entry_count = REQUIRED_ENTRIES - len(cards)
    missing_entries = torch.zeros(
        missing_entry_count * 4,
    )
    serialized_cards = torch.cat((serialized_cards, missing_entries))

    return torch.Tensor(serialized_cards)


def serialize_potions(
    potions: list[Potion], encoding_mapper: EncodingMapper
) -> torch.Tensor:
    serialized_potions = torch.Tensor()
    REQUIRED_ENTRIES = 10

    for i, potion in enumerate(potions):
        this_potion = torch.Tensor(
            [encoding_mapper.get_potion_encoding(potion.name), i]
        )
        serialized_potions = torch.cat((serialized_potions, this_potion))

    missing_entry_count = REQUIRED_ENTRIES - len(potions)
    missing_entries = torch.zeros(
        missing_entry_count * 2,
    )
    serialized_potions = torch.cat((serialized_potions, missing_entries))
    return serialized_potions


def serialize_powers(
    powers: list[Power], encoding_mapper: EncodingMapper
) -> torch.Tensor:
    serialized_powers = torch.Tensor()
    REQUIRED_ENTRIES = 15

    for power in powers:
        this_power = torch.Tensor(
            [encoding_mapper.get_power_encoding(power.power_name), power.amount]
        )
        serialized_powers = torch.cat((serialized_powers, this_power))

    missing_entry_count = REQUIRED_ENTRIES - len(powers)
    missing_entries = torch.zeros(
        missing_entry_count * 2,
    )
    serialized_powers = torch.cat((serialized_powers, missing_entries))
    return torch.Tensor(serialized_powers)


def serialize_monsters(
    monsters: list[Monster], encoding_mapper: EncodingMapper
) -> torch.Tensor:
    serialized_monsters = torch.Tensor()
    REQUIRED_ENTRIES = 10

    for monster in monsters:
        estimated_damage = monster.move_hits * monster.move_adjusted_damage
        power_tensor = serialize_powers(monster.powers, encoding_mapper)

        this_monster = torch.Tensor(
            [
                encoding_mapper.get_monster_encoding(monster.name),
                monster.monster_index,
                monster.max_hp,
                monster.current_hp,
                monster.block,
                estimated_damage,
                monster.intent.value,
            ]
        )
        this_monster = torch.cat((this_monster, power_tensor))
        serialized_monsters = torch.cat((serialized_monsters, this_monster))

    MONSTER_POWER_TENSOR_SIZE = 30
    missing_entry_count = REQUIRED_ENTRIES - len(monsters)
    missing_entries = torch.zeros(
        missing_entry_count * (MONSTER_POWER_TENSOR_SIZE + 7),
    )
    serialized_monsters = torch.cat((serialized_monsters, missing_entries))
    return serialized_monsters


def serialize_orbs(orbs: list[Orb]) -> torch.Tensor:
    def decode_orb(name: str) -> int:
        logging.debug("Orb name debug " + name)
        orb_type = None
        if name == "Orb Slot":
            orb_type = 0
        elif name == "Lightning":
            orb_type = 1
        elif name == "Frost":
            orb_type = 2
        elif name == "Dark":
            orb_type = 3
        elif name == "Plasma":
            orb_type = 4

        return orb_type

    serialized_orbs = torch.Tensor()
    REQUIRED_ENTRIES = 10

    for orb in orbs:
        this_orb = torch.Tensor([decode_orb(orb.name)])
        serialized_orbs = torch.cat((serialized_orbs, this_orb))

    missing_entry_count = REQUIRED_ENTRIES - len(orbs)
    missing_entries = torch.zeros(
        missing_entry_count * 1,
    )
    serialized_orbs = torch.cat((serialized_orbs, missing_entries))
    return serialized_orbs


def serialize_relics(
    relics: list[Relic], encoding_mapper: EncodingMapper
) -> torch.Tensor:
    serialized_relics = torch.Tensor()
    REQUIRED_ENTRIES = 30

    for relic in relics:
        this_relic = torch.Tensor(
            [encoding_mapper.get_relic_encoding(relic.name), relic.counter]
        )
        serialized_relics = torch.cat((serialized_relics, this_relic))

    missing_entry_count = REQUIRED_ENTRIES - len(relics)
    missing_entries = torch.zeros(
        missing_entry_count * 2,
    )
    serialized_relics = torch.cat((serialized_relics, missing_entries))

    return serialized_relics


# Translate game state to NN readable format
def game_state_to_NN_input(
    gameState: Game, encoding_mapper: EncodingMapper
) -> torch.Tensor:
    playerData: Player | None = gameState.player
    if playerData is None:
        raise Exception("Impossible state, missing player data while encoding")

    game_state_tensor = torch.Tensor()

    player_power_tensor = serialize_powers(playerData.powers, encoding_mapper)
    game_state_tensor = torch.cat((game_state_tensor, player_power_tensor))
    logging.debug(
        "game state tensor size after player power " + str(game_state_tensor.size())
    )

    orb_tensor = serialize_orbs(playerData.orbs)
    game_state_tensor = torch.cat((game_state_tensor, orb_tensor))
    logging.debug("game state tensor size after orbs " + str(game_state_tensor.size()))

    player_energy = playerData.energy
    player_block = playerData.block

    relic_tensor = serialize_relics(gameState.relics, encoding_mapper)
    game_state_tensor = torch.cat((game_state_tensor, relic_tensor))
    logging.debug(
        "game state tensor size after relics " + str(game_state_tensor.size())
    )

    potions_tensor = serialize_potions(gameState.potions, encoding_mapper)
    game_state_tensor = torch.cat((game_state_tensor, potions_tensor))
    logging.debug(
        "game state tensor size after potions " + str(game_state_tensor.size())
    )

    player_current_health = gameState.current_hp
    player_max_health = gameState.max_hp

    player_stats_tensor = torch.Tensor(
        [player_energy, player_block, player_current_health, player_max_health]
    )
    game_state_tensor = torch.cat((game_state_tensor, player_stats_tensor))
    logging.debug(
        "game state tensor size after player stats " + str(game_state_tensor.size())
    )

    cards_in_hand = serialize_cards(gameState.hand, encoding_mapper)
    game_state_tensor = torch.cat((game_state_tensor, cards_in_hand))
    logging.debug(
        "game state tensor size after hand cards " + str(game_state_tensor.size())
    )

    discarded_cards = serialize_cards(gameState.discard_pile, encoding_mapper, 50)
    game_state_tensor = torch.cat((game_state_tensor, discarded_cards))
    logging.debug(
        "game state tensor size after discarded cards " + str(game_state_tensor.size())
    )

    exhausted_cards = serialize_cards(gameState.exhaust_pile, encoding_mapper, 25)
    game_state_tensor = torch.cat((game_state_tensor, exhausted_cards))
    logging.debug(
        "game state tensor size after exhausted cards " + str(game_state_tensor.size())
    )

    deck_cards = serialize_cards(gameState.draw_pile, encoding_mapper, 50)
    game_state_tensor = torch.cat((game_state_tensor, deck_cards))
    logging.debug(
        "game state tensor size after deck cards " + str(game_state_tensor.size())
    )

    enemy_tensor = serialize_monsters(gameState.monsters, encoding_mapper)
    game_state_tensor = torch.cat((game_state_tensor, enemy_tensor))
    logging.debug("game state tensor size after enemy " + str(game_state_tensor.size()))

    logging.debug("Encoded state into tensor of length " + str(len(game_state_tensor)))
    return game_state_tensor


# Translate NN output format to readable game state
# Output is is (300,)
def NN_output_to_action(action_index: int) -> Action:
    type = action_index // 100
    action_index = action_index % 100

    using_index = action_index // 10
    action_index = action_index % 10
    target_index = action_index

    if type == 0:
        return EndTurnAction()
    # Card
    elif type == 1:
        return PlayCardAction(card_index=using_index, target_index=target_index)
    # Potion
    elif type == 2:
        return PotionAction(True, potion_index=using_index, target_index=target_index)

    # Todo flag this as an "invalid state" and punish
    return EndTurnAction()
