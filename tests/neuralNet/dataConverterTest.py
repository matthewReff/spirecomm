import unittest

from neuralNet.dataConverter import *
import json


class TestNeuralNetConvert(unittest.TestCase):
    def test_gameData_to_nn_input(self):
        raw_game_data = None
        with open("example_state.json") as f:
            raw_game_data = json.load(f)
        test_game_data = Game.from_json(raw_game_data)
        # Check that conversion doesn't throw
        # TODO re-add encoding mapper
        converted_nn_data = game_state_to_NN_input(test_game_data, encoding_mapper)

    @unittest.skip("Waiting for sample to test")
    def test_nn_out_to_game_action(self):
        testNNData = ""
        returnedAction = NN_output_to_action(testNNData)
        self.assertEqual(type(returnedAction), type(EndTurnAction()))


if __name__ == "__main__":
    unittest.main()
