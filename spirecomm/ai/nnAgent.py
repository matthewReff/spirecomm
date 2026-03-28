from neuralNet.metricLogger import MetricLogger
from utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *
from spirecomm.ai.agent import Agent
from neuralNet.agent import SlayAiAgent
from neuralNet.interactor import NeuralNetInteractor
import datetime
from pathlib import Path


class NnAgent(Agent):
    encoding_mapper = None

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        slay_ai_agent = SlayAiAgent()

        save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S"
        )
        save_dir.mkdir(parents=True)
        training_logger = MetricLogger()
        self.interactor = NeuralNetInteractor(slay_ai_agent, training_logger)

        db = EncodingDatabase(chosen_class)
        db._upsert_tables()
        self.encoding_mapper = EncodingMapper(db)
        super().__init__(chosen_class)

    def change_class(self, chosen_class: PlayerClass):
        db = EncodingDatabase(chosen_class)
        self.encoding_mapper = EncodingMapper(db)
        return super().change_class(chosen_class)

    def before_action_taken(self):
        self.interactor.save_game_state(self.game)
        self.interactor.learn_from_action()

    def get_next_combat_action(self):
        self.encoding_mapper.scrape_state(self.game)

        # You stayed alive
        self.interactor.grant_reward(0.1)

        return self.interactor.run_combat(self.game)

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

        self.interactor.grant_reward(1)
        return super().get_next_combat_reward_action()

    def get_next_boss_reward_action(self):
        self.encoding_mapper.scrape_state(self.game)

        self.interactor.grant_reward(10)
        return super().get_next_boss_reward_action()
