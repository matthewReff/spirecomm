from dataclasses import dataclass


@dataclass
class MetaParameters:
    # Exploration params
    EXPLORATION_RATE_MIN: float  # Lowest possible
    BASE_DECAY_RATE: float  # Multiplier for how fast the model will explore instead of exploit over time

    # Learning Params
    TIME_DISCOUNT_FACTOR: (
        float  # AKA Gamma, what amount the agent will discount future rewards
    )
    LEARNING_RATE: float  # Learning rate passed into model optimizer

    # Memory params
    BURN_IN: int  # min. experiences before training
    LEARN_EVERY: int  # no. of experiences between updates to Q_online
    SYNC_EVERY: int  # no. of experiences between Q_target & Q_online sync

    # Developer Config
    MAX_EPISODES: int  # NOOP
    SAVE_EVERY: int  # no. of steps between state checkpoints being created


@dataclass
class TrainingState:
    CURRENT_EXPLORATION_RATE: (
        float  # Percent chance that agent will  (in float w/ 100% as 1)
    )
    CURRENT_STEP: int  # no. of times the agent has been asked to provide a next move
    CURRENT_EPISODE: int  # no. of total failed/successful runs have been tried
