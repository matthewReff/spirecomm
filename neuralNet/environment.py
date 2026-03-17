from neuralNet.stateDefinitions import ActionSpace, StateSpace
from gymnasium import Env


class SlayAiEnvironment(Env):
    def __init__(self):
        self.action_space = ActionSpace
        self.observation_space = StateSpace
