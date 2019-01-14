# coding: utf-8
# Author: Ernst Dinkelmann

import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, sampling_method='uniform', random_seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            sampling_method (str): each time a sample needs to be drawn from the replay buffer, this parameter
                defines what sampling method is used to draw the sample (one of 'uniform', 'linear' or 'geometric').
                Many are possible, but only three are implemented here.
                    'uniform' - select samples uniformly without replacement.
                    'linear' - select samples with replacement where probabilities are linearly adjusted to make more
                        recent samples more likely (for example).
                    'geometric' - use a geometric distribution to select samples with replacement, making more recent
                        samples more likely.
            random_seed (int): random seed for the numpy and random libraries.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.sampling_method = sampling_method

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add new experience(s) to memory.

        Note the possibility of receiving multiple experiences at the same time,
        due to the possibility of observing multiple agents at the same time.
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        Three different methods implemented. Described in the _init_ part.
        """
        if self.sampling_method == 'uniform':
            experiences = random.sample(self.memory, k=self.batch_size)
        elif self.sampling_method == 'linear':
            experiences = self.sample_linear_batch()
        elif self.sampling_method == 'geometric':
            experiences = self.sample_geometric_batch()
        else:
            raise ValueError('Unknown replay buffer sampling method: ' + self.sampling_method)

        if type(experiences) == 'list':
            states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        else:  # numpy.ndarray  # The linear methods do not retain the namedtuples in a list but the order in an array
            states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
            dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def sample_geometric_int(self, start, end, bias):
        """
        Draws a single sample from a geometric distribution between two values

        Params
        ======
        start (int): value drawn will not be smaller than this
        end (int): value drawn will not be larger than this
        bias (int): geometric distribution parameter. Smaller values implies less skewness.
        """
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def sample_geometric_batch(self, geometric_bias=0.00000001):  # could work, but too slow given the iteration
        """
        Samples, with replacement, a batch, using a geometric distribution, favouring more recent samples.

        Params
        ======
        geometric_bias (float in (0, 1)): geometric distribution parameter. Smaller values implies less skewness.
        """

        experiences = []
        for _ in range(self.batch_size):
            experiences.append(self.memory[self.sample_geometric_int(0, len(self.memory), geometric_bias)])
        return experiences

    def sample_linear_batch(self, max_min_ratio=1.1):  # could work perhaps, but too slow
        """samples, with replacement, a batch, using a linear method, favouring more recent samples

        Params
        ======
        max_min_ratio: probability of drawing the most recent value over the probability of drawing the first value. interpolated for others
        """

        l = len(self.memory)
        # m = (max_min_ratio - 1) / (l - max_min_ratio)
        x = np.array(range(l))
        y = (x + 1) * ((max_min_ratio - 1) / (l - max_min_ratio)) + 1
        y_prob = y / y.sum()

        # Maths
        # (len * m + 1) / (m +1) = max_min_ratio
        # len * m + 1 = max_min_ratio * m + max_min_ratio
        # m * (len - max_min_ratio) = max_min_ratio - 1
        # m = (max_min_ratio - 1) / (len - max_min_ratio)

        idx = (np.interp(np.random.uniform(low=0, high=1, size=self.batch_size), y_prob.cumsum(), x)).astype(int) + 1

        return np.array(self.memory)[idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
