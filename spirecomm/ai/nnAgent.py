from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.communication.action import (
    Action,
    EndTurnAction,
    PlayCardAction,
    PotionAction,
)
from neuralNet.metricLogger import MetricLogger
from utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.spire.character import Monster, Player, PlayerClass
from spirecomm.ai.agent import Agent
from neuralNet.agent import SlayAiAgent
from neuralNet.interactor import NeuralNetInteractor
import datetime
from pathlib import Path
import logging


class NnAgent(Agent):
    encoding_mapper = None
    slay_ai_agent = None
    training_logger = None

    def __init__(self, chosen_class):
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S"
        )
        save_dir.mkdir(parents=True)
        self.log_every = 10

        self.slay_ai_agent = SlayAiAgent(save_dir)

        self.training_logger = MetricLogger(save_dir)

        db = EncodingDatabase(chosen_class)
        db._upsert_tables()
        self.encoding_mapper = EncodingMapper(db)

        self.interactor = NeuralNetInteractor(
            self.slay_ai_agent, self.training_logger, self.encoding_mapper
        )
        super().__init__(chosen_class)

    def change_class(self, chosen_class: PlayerClass):
        db = EncodingDatabase(chosen_class)
        db._upsert_tables()
        self.encoding_mapper = EncodingMapper(db)
        self.interactor = NeuralNetInteractor(
            self.slay_ai_agent, self.training_logger, self.encoding_mapper
        )
        return super().change_class(chosen_class)

    def before_combat_action(self):
        self.encoding_mapper.scrape_state(self.game)
        self.interactor.save_game_state(self.game)
        self.interactor.learn_from_action()

    def after_game_end(self):
        logging.debug("after_game_end called")
        self.training_logger.log_episode()

        # Log episode stuff sometimes
        current_episode = self.slay_ai_agent.curr_episode
        if (current_episode % self.log_every == 0) or (
            current_episode == self.slay_ai_agent.max_episodes - 1
        ):
            self.training_logger.record(
                episode=current_episode,
                epsilon=self.slay_ai_agent.exploration_rate,
                step=self.slay_ai_agent.curr_step,
            )
        self.slay_ai_agent.curr_episode = self.slay_ai_agent.curr_episode + 1

    def normalize_combat_action(self, raw_action: Action) -> Action:
        if raw_action.command == "end":
            logging.debug("Got end turn action to normalize")
            return raw_action

        is_valid_source = None
        monster_index = None
        if raw_action.command == "play":
            card_action: PlayCardAction = raw_action
            card_index = card_action.card_index
            monster_index = card_action.target_index
            hand_cards = self.game.hand
            logging.debug(
                "Got card action to normalize:"
                + str(card_index)
                + "/"
                + str(len(hand_cards))
            )

            if (len(hand_cards) - 1) < card_index:
                is_valid_source = False
            else:
                actual_card: Card = hand_cards[card_index]
                player: Player = self.game.player
                current_player_energy = player.energy

                is_valid_source = (
                    current_player_energy >= actual_card.cost
                    and actual_card.is_playable
                )

                # No required target, bypass filtering
                if not actual_card.has_target and is_valid_source:
                    return PlayCardAction(card_index=card_action.card_index)

        if raw_action.command == "potion":
            potion_action: PotionAction = raw_action
            potion_index = potion_action.potion_index
            monster_index = potion_action.target_index
            potions = self.game.potions
            logging.debug(
                "Got potion action to normalize: "
                + str(potion_index)
                + "/"
                + str(len(potions))
            )

            if (len(potions) - 1) < potion_index:
                is_valid_source = False
            else:
                actual_potion: Potion = potions[potion_index]
                is_valid_source = actual_potion.can_use

                # No required target, bypass filtering
                if not actual_potion.requires_target and is_valid_source:
                    return PotionAction(True, potion_index=potion_action.potion_index)

        is_valid_target = None
        monsters = self.game.monsters
        logging.debug(
            "Got Monster target to normalize:"
            + str(monster_index)
            + "/"
            + str(len(self.game.monsters))
        )
        if (len(monsters) - 1) < monster_index:
            is_valid_target = False
        else:
            actual_monster: Monster = monsters[monster_index]
            is_valid_target = not actual_monster.is_gone

        is_invalid_action = not is_valid_target or not is_valid_source
        if is_invalid_action:
            # Took an impossible action, not cool buddy
            # self.interactor.grant_reward(-0.5)

            return EndTurnAction()

        return raw_action

    def get_next_combat_action(self) -> Action:
        raw_action = self.interactor.run_combat(self.game)

        # You stayed alive, that's nice. But you need to actually DO something
        self.interactor.grant_reward(-0.1)

        return self.normalize_combat_action(raw_action)

    def get_card_reward_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_card_reward_action()

    def get_rest_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_rest_action()

    def get_screen_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_screen_action()

    def get_map_choice_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_map_choice_action()

    def get_next_combat_reward_action(self):
        self.encoding_mapper.scrape_state(self.game)

        # Winning a combat is good
        self.interactor.grant_reward(0.1)
        return super().get_next_combat_reward_action()

    def get_next_boss_reward_action(self):
        self.encoding_mapper.scrape_state(self.game)

        # Killing a boss is very good
        self.interactor.grant_reward(1)
        return super().get_next_boss_reward_action()
