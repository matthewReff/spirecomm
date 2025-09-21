from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from neuralNet.dataConverter import *


class NeuralNetInteractor:
    def __init__(self):
        pass

    def __run_combat_in_NN(self, nnCombatIpnut: str) -> str:
        return ""

    def run_combat(self, gameState: Game) -> Action:
        readableState = game_state_to_NN_input(gameState)

        nnOutput = self.__run_combat_in_NN(readableState)

        return NN_output_to_action(nnOutput)
