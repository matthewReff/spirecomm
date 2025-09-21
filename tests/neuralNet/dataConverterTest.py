import unittest

from neuralNet.dataConverter import *


class TestNeuralNetConvert(unittest.TestCase):
    def test_gameData_to_nn_input(self):
        testGameData = Game()
        convertedNNData = game_state_to_NN_input(testGameData)
        self.assertEqual(convertedNNData, "")

    def test_nn_out_to_game_action(self):
        testNNData = ""
        returnedAction = NN_output_to_action(testNNData)
        self.assertEqual(type(returnedAction), type(EndTurnAction()))


if __name__ == "__main__":
    unittest.main()
