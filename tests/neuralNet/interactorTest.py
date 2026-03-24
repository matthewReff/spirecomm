import unittest

from neuralNet.agent import SlayAiAgent
from neuralNet.interactor import NeuralNetInteractor


class TestInteractorCreation(unittest.TestCase):
    def test_can_create_from_scratch(self):
        slay_ai_agent = SlayAiAgent()
        interactor = NeuralNetInteractor(slay_ai_agent)


if __name__ == "__main__":
    unittest.main()
