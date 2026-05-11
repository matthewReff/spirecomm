from pathlib import Path
import random

from neuralNet.configurationData import MetaParameters, TrainingState
from neuralNet.dataConverter import ACTION_ENCODING
from spirecomm.spire.game import Game
from neuralNet.network import SlayAiNet
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage
import logging


class SlayAiAgent:
    @staticmethod
    def generate_default_parameters(speedup_factor: int) -> MetaParameters:
        DEFAULT_EXPLORATION_RATE_MIN = 0.1
        DEFAULT_DECAY_RATE = 0.99999975

        # Given a speedup factor, reach the exploration min N times faster
        NUMBER_OF_STEPS_TO_REACH_ORIGINAL = np.log(
            DEFAULT_EXPLORATION_RATE_MIN
        ) / np.log(DEFAULT_DECAY_RATE)
        NUMBER_OF_STEPS_TO_REACH_SPED_UP = (
            NUMBER_OF_STEPS_TO_REACH_ORIGINAL / speedup_factor
        )
        SCALED_DECAY_RATE = np.pow(
            DEFAULT_EXPLORATION_RATE_MIN, 1 / NUMBER_OF_STEPS_TO_REACH_SPED_UP
        )

        return MetaParameters(
            EXPLORATION_RATE_MIN=DEFAULT_EXPLORATION_RATE_MIN,
            BASE_DECAY_RATE=SCALED_DECAY_RATE,
            TIME_DISCOUNT_FACTOR=0.99,
            LEARNING_RATE=0.00025,
            MAX_EPISODES=999999,
            SAVE_EVERY=5e5 // speedup_factor,
            BURN_IN=1e4 // speedup_factor,
            LEARN_EVERY=3,
            SYNC_EVERY=1e4 // speedup_factor,
        )

    @staticmethod
    def generate_staring_state() -> TrainingState:
        return TrainingState(
            CURRENT_STEP=0, CURRENT_EPISODE=0, CURRENT_EXPLORATION_RATE=1
        )

    def __init__(
        self,
        save_dir: Path,
        network_state_dict,
        optimizer_state_dict,
        memory_state_dict,
        meta_params: MetaParameters | None,
        training_state: TrainingState | None,
    ):
        # Generic Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = 32
        self.state_size = 1034
        self.action_size = 300

        self.starting_state = (
            training_state
            if training_state is not None
            else SlayAiAgent.generate_staring_state()
        )
        self.curr_step = self.starting_state.CURRENT_STEP
        self.curr_episode = self.starting_state.CURRENT_EPISODE
        self.exploration_rate = self.starting_state.CURRENT_EXPLORATION_RATE

        self.meta_params = (
            meta_params
            if meta_params is not None
            else SlayAiAgent.generate_default_parameters(speedup_factor=1e1)
        )
        self.exploration_rate_decay = self.meta_params.BASE_DECAY_RATE
        self.exploration_rate_min = self.meta_params.EXPLORATION_RATE_MIN
        self.max_episodes = self.meta_params.MAX_EPISODES
        self.burnin = self.meta_params.BURN_IN
        self.save_every = self.meta_params.SAVE_EVERY
        self.sync_every = self.meta_params.SYNC_EVERY
        self.learn_every = self.meta_params.LEARN_EVERY

        self.net = SlayAiNet(
            save_dir=save_dir,
            batch_size=self.batch_size,
            state_size=self.state_size,
            action_size=self.action_size,
            params=self.meta_params,
            network_state_dict=network_state_dict,
            optimizer_state_dict=optimizer_state_dict,
        ).float()
        self.net = self.net.to(device=self.device)

        self.memory = TensorDictReplayBuffer(
            storage=ListStorage(100000, device=torch.device("cpu"))
        )
        if memory_state_dict is not None:
            self.memory.load_state_dict(memory_state_dict)

    def _true_random_action(self) -> int:
        random_using_index = random.randint(0, 9)
        random_target_index = random.randint(0, 9)

        # 10% change of ending turn, 10% chance of potion, 80% chance of using card
        action_options = [
            ACTION_ENCODING.END_TURN,
            ACTION_ENCODING.USE_POTION,
            *(ACTION_ENCODING.PLAY_CARD for _ in range(8)),
        ]
        action_type = random.choice(action_options)

        encodedIndex = (
            (action_type.value * 100) + (random_using_index * 10) + random_target_index
        )
        logging.debug("Taking a true random action " + str(encodedIndex))
        return encodedIndex

    # Put on some training wheels and at least try to play actual cards
    def assisted_random_action(self, game_state: Game) -> int:
        monster_count = len(game_state.monsters)
        # Intentionally don't filter down to just playable potions
        potion_count = len(game_state.potions)
        # Intentionally don't filter down to just playable cards
        card_count = len(game_state.hand)

        # 10% change of ending turn, 10% chance of potion, 80% chance of using card
        action_options = [
            ACTION_ENCODING.END_TURN,
            ACTION_ENCODING.USE_POTION,
            *(ACTION_ENCODING.PLAY_CARD for _ in range(8)),
        ]
        action_type = random.choice(action_options)

        using_index = 0
        target_index = random.randrange(0, monster_count)
        if action_type == ACTION_ENCODING.PLAY_CARD:
            using_index = random.randrange(0, card_count)

        if action_type == ACTION_ENCODING.USE_POTION:
            using_index = random.randrange(0, potion_count)

        encodedIndex = (action_type.value * 100) + (using_index * 10) + target_index
        logging.debug("Taking an assisted random action " + str(encodedIndex))
        return encodedIndex

    def optimalAction(self, game_state: torch.Tensor) -> int:
        game_state = torch.tensor(game_state, device=self.device).unsqueeze(0)
        action_values = self.net(game_state, model="online")
        action_index = torch.argmax(action_values, axis=1).item()

        logging.debug("Determining an optimal action " + str(action_index))
        return action_index

    def act(self, encoded_game_state: torch.Tensor, raw_game_state: Game):
        should_explore = np.random.rand() < self.exploration_rate

        if should_explore:
            action_index = self.assisted_random_action(raw_game_state)
        else:
            action_index = self.optimalAction(encoded_game_state)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_index

    # Make sure to encode the state before caching
    def cache(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: int,
        reward: int,
        done: int,
    ):
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        logging.info("Reward during cache " + str(reward))

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.net.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            current_training_state = TrainingState(
                CURRENT_STEP=self.curr_step,
                CURRENT_EPISODE=self.curr_episode,
                CURRENT_EXPLORATION_RATE=self.exploration_rate,
            )

            self.net.save(
                curr_step=self.curr_step,
                save_every=self.save_every,
                memory=self.memory,
                params=self.meta_params,
                training_state=current_training_state,
            )

        if self.curr_step < self.burnin:
            logging.debug(
                "Current step is "
                + str(self.curr_step)
                + ", will start learning after "
                + str(self.burnin)
            )
            return None, None

        if self.curr_step % self.learn_every != 0:
            logging.debug(
                "Current step is "
                + str(self.curr_step)
                + ", skipping since step is not a multiple of "
                + str(self.learn_every)
            )
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.net.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.net.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.net.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
