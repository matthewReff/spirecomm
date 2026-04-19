from pathlib import Path

import torch
import numpy as np
from torch import nn
import logging

# https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


class SlayAiNet(nn.Module):
    save_dir = None

    def __init__(
        self, save_dir: Path, batch_size: int, state_size: int, action_size: int
    ):
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.online = self.__build_nn(self.state_size, self.action_size)

        self.target = self.__build_nn(self.state_size, self.action_size)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.gamma = 0.9
        self.learning_rate = 0.00025
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

    def save(self, curr_step: int, save_every: int, exploration_rate: int):
        save_path = self.save_dir / f"slay_ai_net_{int(curr_step // save_every)}.chkpt"
        torch.save(
            dict(model=self.online.state_dict(), exploration_rate=exploration_rate),
            save_path,
        )
        logging.info(f"SlayAiNet saved to {save_path} at step {curr_step}")
