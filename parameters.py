# General parameters
RANDOM_SEED = 0  # Global random seed ensuring reproducible results 0
MAX_N_EPISODES = 10000  # Overall max number of training episodes
MAX_N_STEPS = 1001  # Maximum number of training steps per episode before starting a new episode
WIN_SCORE = 0.5  # Project goal. Once reached, will save critic and actor networks' weights to checkpoint files
WIN_QUEUE_LEN = 100  # The number of episodes over which the total reward is averaged to get score

# Agent parameters
SHARED_REPLAY_BUFFER = True  # Whether the agents share a single replay buffer or not.
REPLAY_BUFFER_SIZE = int(1e5)  # replay buffer size int(1e5)
REPLAY_BUFFER_SAMPLING_METHOD = 'uniform'  # one of 'uniform', 'geometric' or 'linear'. refer replay_buffer.py for details
BATCH_SIZE = 1024  # mini-batch size 1024
GAMMA = 0.9999  # discount factor for rewards 0.9999
TAU = 1e-3  # for soft update of target parameters 1e-3
LR_ACTOR = 2e-5  # learning rate of the actor 2e-5
LR_CRITIC = 2e-4  # learning rate of the critic 2e-4
LR_DECAY_99_EVERY_ACTOR = 1e10  # Optimizer gets re-declared with a Learning Rate *= 0.99 every x number of steps (stability) 5e4
LR_DECAY_99_EVERY_CRITIC = 1e10  # Optimizer gets re-declared with a Learning Rate *= 0.99 every x number of steps (stability) 5e4
WEIGHT_DECAY_ACTOR = 0  # L2 weight decay 1e-4  0
WEIGHT_DECAY_CRITIC = 0  # L2 weight decay 1e-4 0
USE_BATCH_NORM = False  # whether to use batch normalisation
UPDATE_EVERY = 1  # Update weights this often in steps
CLIP_GRADIENT_ACTOR = True  # Whether to clip the Actor Network's Parameter's Gradient to a norm of 1.
CLIP_GRADIENT_CRITIC = True  # Whether to clip the Critic Network's Parameter's Gradient to a norm of 1

# Parameters for noise added to actions
ADD_ACTION_NOISE_TRAINING = True  # Whether to add noise to the actions in training. Not using noise in viewing mode
ACTION_NOISE_METHOD = 'adjusted'  # one of 'initial' or 'adjusted'. Refer noise.py for details
NOISE_SIGMA = 1  # initial variability 1
NOISE_THETA = 1  # extent of continuing in a direction (like a momentum of sorts) 1
NOISE_DT = 1  # extent of short-term variability 1
NOISE_SIGMA_DELTA = 0.9999  # overall variance reduction factor 0.9999

# Network layer parameters
FC1_UNITS = 1024  # Number of fully connected units in the 1st hidden layer, both actor and critic networks 512
FC2_UNITS = 1024  # Number of fully connected units in the 2nd hidden layer, both actor and critic networks 512
FC3_UNITS = 1024  # Number of fully connected units in the 3rd hidden layer, both actor and critic networks 512

# ToDo - even more units ?
