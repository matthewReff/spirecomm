import torch
import numpy as np
from torch import nn


# https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


class SlayAiNet(nn.Module):
    """mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self):
        super().__init__()
        self.online = self.__build_nn()

        self.target = self.__build_nn()
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_nn(self):
        return nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.Linear(512, 299),
        )

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
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
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir
            / f"slay_ai_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"SlayAiNet saved to {save_path} at step {self.curr_step}")
