from pathlib import Path

from neuralNet.configurationData import MetaParameters, TrainingState
import torch
import numpy as np
from torch import nn
import torch.serialization
import logging
from torchrl.data import TensorDictReplayBuffer
import json

torch.serialization.add_safe_globals(
    [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
    ]
)

# https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


class SlayAiNet(nn.Module):
    save_dir = None
    CHECKPOINT_FILENAME = "slay_ai.chkpt"
    PARAMS_FILENAME = "params.json"
    STATE_FILENAME = "state.json"

    def __init__(
        self,
        save_dir: Path,
        batch_size: int,
        state_size: int,
        action_size: int,
        params: MetaParameters,
        network_state_dict,
        optimizer_state_dict,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = params.TIME_DISCOUNT_FACTOR
        self.learning_rate = params.LEARNING_RATE

        self.online = self.__build_nn(self.state_size, self.action_size)
        if network_state_dict is not None:
            self.online.load_state_dict(network_state_dict)

        self.target = self.__build_nn(self.state_size, self.action_size)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.learning_rate
        )
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
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

    @staticmethod
    def load(full_save_dir: str | None):
        if full_save_dir is None:
            return (None, None, None, None, None)

        checkpoint_path = full_save_dir / SlayAiNet.CHECKPOINT_FILENAME
        checkpoint = torch.load(checkpoint_path)

        parameters_path = full_save_dir / SlayAiNet.PARAMS_FILENAME
        params: MetaParameters | None = None
        with open(parameters_path, "r") as file:
            params = MetaParameters(**json.load(file))

        state_path = full_save_dir / SlayAiNet.STATE_FILENAME
        state: TrainingState | None = None
        with open(state_path, "r") as file:
            state = TrainingState(**json.load(file))

        return (
            checkpoint.get("network"),
            checkpoint.get("optimizer"),
            checkpoint.get("memory"),
            params,
            state,
        )

    def save(
        self,
        curr_step: int,
        save_every: int,
        memory: TensorDictReplayBuffer,
        params: MetaParameters,
        training_state: TrainingState,
    ):
        save_slice = self.save_dir / f"{int(curr_step // save_every)}"
        Path(save_slice).mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_slice / SlayAiNet.CHECKPOINT_FILENAME
        torch.save(
            dict(
                network=self.online.state_dict(),
                optimizer=self.optimizer.state_dict(),
                memory=memory.state_dict(),
            ),
            checkpoint_path,
        )
        logging.info(f"SlayAi Data saved to {checkpoint_path} at step {curr_step}")

        parameters_path = save_slice / SlayAiNet.PARAMS_FILENAME
        with open(parameters_path, "w") as file:
            json.dump(params.__dict__, file)

        state_path = save_slice / SlayAiNet.STATE_FILENAME
        with open(state_path, "w") as file:
            json.dump(training_state.__dict__, file)
