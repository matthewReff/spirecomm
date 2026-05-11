from pathlib import Path

from neuralNet.configurationData import MetaParameters, TrainingState
import torch
import numpy as np
from torch import nn
import logging
from torchrl.data import TensorDictReplayBuffer
import json

# https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


class SlayAiNet(nn.Module):
    save_dir = None

    def __init__(
        self,
        save_dir: Path,
        batch_size: int,
        state_size: int,
        action_size: int,
        params: MetaParameters,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = params.TIME_DISCOUNT_FACTOR
        self.learning_rate = params.LEARNING_RATE

        self.online = self.__build_nn(self.state_size, self.action_size)

        self.target = self.__build_nn(self.state_size, self.action_size)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.learning_rate
        )
        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, input, model):
        logging.debug("Doing forward prop " + str(input.size()))
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_nn(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, action_dim),
        )

    def td_estimate(self, state, action):
        logging.debug("td_estimate " + str(action) + " " + str(action.size()))
        current_Q = self(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(
        self,
        curr_step: int,
        save_every: int,
        exploration_rate: int,
        memory: TensorDictReplayBuffer,
        params: MetaParameters,
        training_state: TrainingState,
    ):
        save_slice = self.save_dir / f"int({curr_step // save_every})"

        checkpoint_path = save_slice / "slay_ai_net.chkpt"
        torch.save(
            dict(model=self.online.state_dict(), exploration_rate=exploration_rate),
            checkpoint_path,
        )
        logging.info(f"SlayAiNet saved to {checkpoint_path} at step {curr_step}")

        optimizer_path = save_slice / "optimizer.pt"
        torch.save(dict(model=self.optimizer.state_dict()), optimizer_path)

        memory_path = save_slice / "memory.dat"
        memory.dumps(memory_path)

        parameters_path = save_slice / "params.json"
        with open(parameters_path, "w") as file:
            json.dump(params, file)

        state_path = save_slice / "state.json"
        with open(state_path, "w") as file:
            json.dump(training_state, file)
