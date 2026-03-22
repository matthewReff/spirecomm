from Mods.spirecomm.utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *
from spirecomm.ai.agent import Agent
from neuralNet.agent import SlayAiAgent
from neuralNet.environment import SlayAiEnvironment
from neuralNet.interactor import NeuralNetInteractor


class NnAgent(Agent):
    encoding_mapper = None

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        self.state_dim = (1, 1)
        self.action_dim = (1, 3)
        slay_ai_agent = SlayAiAgent(self.state_dim, self.action_dim)
        slay_ai_environment = SlayAiEnvironment()
        self.interactor = NeuralNetInteractor(slay_ai_agent, slay_ai_environment)

        db = EncodingDatabase(chosen_class)
        self.encoding_mapper = EncodingMapper(db)
        super().__init__(chosen_class)

    def change_class(self, chosen_class: PlayerClass):
        db = EncodingDatabase(chosen_class)
        self.encoding_mapper = EncodingMapper(db)
        return super().change_class(chosen_class)

    def get_next_combat_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_next_combat_action()
        # return self.interactor.run_combat(self.game)

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
        return super().get_next_combat_reward_action()

    def get_next_boss_reward_action(self):
        self.encoding_mapper.scrape_state(self.game)
        return super().get_next_boss_reward_action()
