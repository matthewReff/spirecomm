import unittest

from neuralNet.interactor import *
from spirecomm.communication.action import *


class TestNeuralNetConvert(unittest.TestCase):
    def test_run_combat_returns_action(self):
        dummyState = Game()
        interactor = NeuralNetInteractor()

        returnedAction = interactor.run_combat(dummyState)
        self.assertEqual(type(returnedAction), type(EndTurnAction()))


if __name__ == "__main__":
    unittest.main()
