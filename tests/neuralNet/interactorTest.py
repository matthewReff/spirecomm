import unittest

from neuralNet.agent import SlayAiAgent
from neuralNet.environment import SlayAiEnvironment
from neuralNet.interactor import NeuralNetInteractor


class TestInteractorCreation(unittest.TestCase):
    def test_can_create_from_scratch(self):
        slay_ai_agent = SlayAiAgent(self.state_dim, self.action_dim)
        slay_ai_environment = SlayAiEnvironment()
        interactor = NeuralNetInteractor(slay_ai_agent, slay_ai_environment)


if __name__ == "__main__":
    unittest.main()
