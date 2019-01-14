# coding: utf-8
# Author: Ernst Dinkelmann

import numpy as np
import random
import copy


class InitialOrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process.

    This was actually the one found in a number of implementations.
    It does not actually look like Ornstein-Uhlenbeck to me from sources on the internet.
    It also only gives positive noise, which I am sure is not actually what we want to do to explore.

    I'm not summarising the parameters - these can be found on the internet and the code below is very simple to
    understand as well.
    """

    def __init__(self, shape, random_seed, x0, mu, sigma=0.2, theta=0.15):
        """Initialize parameters and noise process."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.shape = shape
        self.x0 = x0 * np.ones(shape)
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.x0)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(0, 1, size=self.shape)
        self.state = x + dx
        return self.state


class AdjustedOrnsteinUhlenbeckActionNoise:
    """
    Adjusted Decreasing Variance Ornstein-Uhlenbeck process.

    This is (as far as I understand it) a correct implementation of a Ornstein-Uhlenbeck process.
    It is adjusted in one aspect only - the addition of a sigma_delta parameter that allows changing of the
    sigma parameter with each call to sample. In the case of action noise, this is ideal - we can start with
    a larger value for sigma to encourage more random exploration and reduce over time to focus around the optimal
    trajectories.

    I'm not summarising the parameters - these can be found on the internet and the code below is very simple to
    understand as well.
    """

    def __init__(self, shape, x0, mu, sigma, theta, dt, sigma_delta=0.99997, random_seed=0):
        """Initialize parameters and noise process."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.shape = shape
        self.x0 = x0 * np.ones(shape)
        self.mu = mu * np.ones(shape)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.sigma_delta = sigma_delta
        self.reset()

    def reset(self):
        self.x_prev = copy.copy(self.x0)

    def sample(self):
        self.x_prev += self.theta * (self.mu - self.x_prev) * self.dt + \
                       self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.shape)
        self.sigma *= self.sigma_delta  # reduce variance
        return self.x_prev
