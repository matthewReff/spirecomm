import unittest

from spirecomm.spire.character import PlayerClass
from utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.communication.action import EndTurnAction
from spirecomm.spire.game import Game
from neuralNet.dataConverter import game_state_to_NN_input, NN_output_to_action
import json
import os


class TestNeuralNetConvert(unittest.TestCase):
    def test_gameData_to_nn_input(self):
        raw_game_data = None

        current_directory = os.path.dirname(os.path.realpath(__file__))
        state_file_path = os.path.join(current_directory, "example_state.json")

        with open(state_file_path) as f:
            raw_game_data = json.load(f)
        test_game_data = Game.from_json(
            raw_game_data["game_state"], raw_game_data["available_commands"]
        )

        db = EncodingDatabase(PlayerClass.THE_SILENT)
        encoding_mapper = EncodingMapper(db)

        encoding_mapper.scrape_state(test_game_data)

        converted_nn_data = game_state_to_NN_input(test_game_data, encoding_mapper)

    def test_nn_out_to_game_action(self):
        test_nn_action = 1
        returnedAction = NN_output_to_action(test_nn_action)
        self.assertEqual(type(returnedAction), type(EndTurnAction()))


if __name__ == "__main__":
    unittest.main()
