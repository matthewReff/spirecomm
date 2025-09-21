from spirecomm.spire.game import Game
from spirecomm.communication.action import *


# Translate game state to NN readable format
def game_state_to_NN_input(gameState: Game) -> str:
    return ""


# Translate NN output format to readable game state
def NN_output_to_action(networkOutput: str) -> Action:
    return EndTurnAction()
