import unittest

from neuralNet.dataConverter import *


class TestNeuralNetConvert(unittest.TestCase):
    def test_gameData_to_nn_input(self):
        testGameData = Game()
        # Check that conversion doesn't throw
        convertedNNData = game_state_to_NN_input(testGameData)

    @unittest.skip("Waiting for sample to test")
    def test_nn_out_to_game_action(self):
        testNNData = ""
        returnedAction = NN_output_to_action(testNNData)
        self.assertEqual(type(returnedAction), type(EndTurnAction()))


if __name__ == "__main__":
    unittest.main()
