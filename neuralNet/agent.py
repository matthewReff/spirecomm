from pathlib import Path
import random

from neuralNet.network import SlayAiNet
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage
import logging


class SlayAiAgent:
    def __init__(self, save_dir: Path):
        # Generic Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = 32

        self.net = SlayAiNet(save_dir, self.batch_size).float()
        self.net = self.net.to(device=self.device)

        # Exploration params
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.curr_episode = 0
        self.max_episodes = 999999  # TODO add actual stopping after specific episode

        # Memory params
        # self.save_every = 5e5
        self.save_every = 1e1
        self.memory = TensorDictReplayBuffer(
            storage=ListStorage(100000, device=torch.device("cpu"))
        )

        # self.burnin = 1e4  # min. experiences before training
        self.burnin = 1e1  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        # self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.sync_every = 1e1  # no. of experiences between Q_target & Q_online sync

    def randomAction(self) -> int:
        random_action_number = random.randint(0, 10)
        random_using_index = random.randint(0, 9)
        random_target_index = random.randint(0, 9)

        # 10% change of ending turn, 10% chance of potion, 80% chance of using card
        action_type = 0
        if random_action_number == 1:
            action_type = 2
        elif 1 < random_action_number <= 9:
            action_type = 1
        else:
            pass

        encodedIndex = (
            (action_type * 100) + (random_using_index * 10) + random_target_index
        )
        return encodedIndex

    def optimalAction(self, game_state: torch.Tensor) -> int:
        game_state = (
            game_state[0].__array__()
            if isinstance(game_state, tuple)
            else game_state.__array__()
        )
        game_state = torch.tensor(game_state, device=self.device).unsqueeze(0)
        action_values = self.net(game_state, model="online")
        action_index = torch.argmax(action_values, axis=1).item()

        return action_index

    def act(self, game_state):
        should_explore = np.random.rand() < self.exploration_rate

        if should_explore:
            action_index = self.randomAction()
        else:
            action_index = self.optimalAction(game_state)

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

        logging.debug("state tensor " + str(state.size()))
        logging.debug("next_state tensor " + str(next_state.size()))
        logging.debug("action tensor " + str(action) + " " + str(action.size()))
        logging.debug("reward tensor " + str(reward.size()))
        logging.debug("done tensor " + str(done.size()))

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
            self.net.save(
                curr_step=self.curr_step,
                save_every=self.save_every,
                exploration_rate=self.exploration_rate,
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
