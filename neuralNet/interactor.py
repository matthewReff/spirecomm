from spirecomm.spire.game import Game
from spirecomm.communication.action import *
from neuralNet.dataConverter import *


class NeuralNetInteractor:
    def __init__(self, aiAgent):
        self.aiAgent = aiAgent

    def run_combat(self, gameState: Game) -> Action:
        readableState = game_state_to_NN_input(gameState)

        nnAction = self.aiAgent.act(readableState)

        return NN_output_to_action(nnAction)
