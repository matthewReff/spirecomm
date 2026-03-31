from utilities.sqlite_scraping import EncodingMapper
from neuralNet.agent import SlayAiAgent
from neuralNet.metricLogger import MetricLogger
from spirecomm.spire.game import Game
from spirecomm.communication.action import Action
from neuralNet.dataConverter import NN_output_to_action, game_state_to_NN_input


class NeuralNetInteractor:
    ai_agent = None
    metrics_logger = None
    encoding_mapper = None

    current_game_state = None
    last_game_state = None
    last_action = None
    reward_since_last = 0

    def __init__(
        self,
        ai_agent: SlayAiAgent,
        metrics_logger: MetricLogger,
        encoding_mapper: EncodingMapper,
    ):
        self.ai_agent = ai_agent
        self.metrics_logger = metrics_logger
        self.encoding_mapper = encoding_mapper

    def run_combat(self, gameState: Game) -> Action:
        nn_state = game_state_to_NN_input(gameState, self.encoding_mapper)

        nnAction = self.ai_agent.act(nn_state)
        self.last_action = nnAction

        return NN_output_to_action(nnAction)

    def save_game_state(self, game_state: Game):
        self.last_game_state = self.current_game_state
        self.current_game_state = game_state

    def learn_from_action(self):
        if self.last_game_state is None or self.current_game_state is None:
            return

        self.ai_agent.cache(
            game_state_to_NN_input(self.last_game_state, self.encoding_mapper),
            game_state_to_NN_input(self.current_game_state, self.encoding_mapper),
            self.last_action,
            self.reward_since_last,
            0,
        )  # TODO fix done attribute

        # Learn
        q, loss = self.ai_agent.learn()

        # Logging
        self.metrics_logger.log_step(self.reward_since_last, loss, q)

        self.reset_reward()

    def reset_reward(self):
        self.reward_since_last = 0

    def grant_reward(self, reward_amount: int):
        self.reward_since_last = self.reward_since_last + reward_amount
