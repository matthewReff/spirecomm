
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *
from spirecomm.ai.agent import Agent


class SimpleAgent(Agent):
    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        super().__init__(chosen_class)

    def change_class(self, chosen_class: PlayerClass):
        return super().change_class(chosen_class)

    def get_next_combat_action(self):
        return super().get_next_combat_action()

    def get_card_reward_action(self):
        return super().get_card_reward_action()

    def get_rest_action(self):
        return super().get_rest_action()

    def get_screen_action(self):
        return super().get_screen_action()

    def get_map_choice_action(self):
        return super().get_map_choice_action()

    def get_next_combat_reward_action(self):
        return super().get_next_combat_reward_action()

    def get_next_boss_reward_action(self):
        return super().get_next_boss_reward_action()
