# coding: utf-8
# Author: Ernst Dinkelmann

import numpy as np
import random

from parameters import *
from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from noise import InitialOrnsteinUhlenbeckActionNoise, AdjustedOrnsteinUhlenbeckActionNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiDdpgAgent:
    """
        A Multi Agent co-ordinating multiple Deep Deterministic Policy Gradient agents
        It's purpose is co-ordination of agents. It is not an agent that learns it's own weights in a network.
        It receives the obervations and sends these on to the agents in an appropriate manner.
        It receives information from the agents, which it communicates with the environment as appropriate.

        """
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """
        Initialize an Multi Agent object.

        Params
        ======
            num_agents (int): number of agents to be co-ordinated. multiple agents are handled within the class.
            state_size (int): dimension of each state per agent
            action_size (int): dimension of each action per agent
            random_seed (int): random seed
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.t_step = 0  # A counter that increases each time the "step" function is executed
        self.state_size = state_size
        self.action_size = action_size

        # Shared Replay Buffer
        # If the global parameter indicates that we should use a shared replay buffer, we create an instance of the
        # ReplayBuffer Class as a class variable. This is important to do as it allows multiple agents to interact
        # with the exact same instance of the class and hence the same memory component.
        if SHARED_REPLAY_BUFFER:
            shared_replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE,
                                                batch_size=BATCH_SIZE,
                                                sampling_method=REPLAY_BUFFER_SAMPLING_METHOD,
                                                random_seed=RANDOM_SEED)
            self.shared_replay_buffer = shared_replay_buffer
        else:
            shared_replay_buffer = None
            self.shared_replay_buffer = []

        # Create a number of individual separate agents in a list, each of which will interact with the environment, act in it and learn from it
        self.ddpg_agents = [DdpgAgent(num_agents=1,   # each agent will have his own class
                                      state_size=state_size,
                                      action_size=action_size,
                                      random_seed=random_seed,
                                      shared_replay_buffer=shared_replay_buffer) for _ in range(num_agents)]

    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()

    def act(self, states_all, add_action_noise=False):
        """
        Get actions from all agents. It merely passes the state for each agent, to that agent and gets the agent' action back.

        params
        ======
            states_all (np.array): dimensions [num_agents, num_observations) All agents' state observations
            add_action_noise (bool): Whether to add noise to the returned action
        """
        actions = np.array([np.squeeze(agent.act(np.expand_dims(states, axis=0), add_action_noise=add_action_noise), axis=0)
                            for agent, states in zip(self.ddpg_agents, states_all)])
        return actions

    def step(self, states_all, actions_all, rewards_all, next_states_all, dones_all):
        """Pass experience to the agents step functions so they may store it, and potentially learn from it. Decided by the agent."""
        for agent, states, actions, rewards, next_states, dones in zip(self.ddpg_agents, states_all, actions_all, rewards_all, next_states_all, dones_all):
            agent.step(np.expand_dims(states, axis=0),
                       np.expand_dims(actions, axis=0),
                       np.expand_dims(rewards, axis=0),
                       np.expand_dims(next_states, axis=0),
                       np.expand_dims(dones, axis=0),
                       )


class DdpgAgent:
    """
    A Deep Deterministic Policy Gradient Agent.
    Interacts with and learns from the environment.

    Although the Class allows the observation of multiple agents at the same time and learning from their experiences,
     it is a single agent that is learning (only one actor and one critic network has weights learnt). This is
     different to the case of a true multi agent setup, which is why the MultiDdpgAgent Class is needed. It allows
     separate networks to be trained.

    """
    
    def __init__(self, num_agents, state_size, action_size, random_seed, shared_replay_buffer):
        """
        Initialize an Agent object.
        
        Params
        ======
            num_agents (int): number of agents observed at the same time. multiple agents observations are handled within the class.
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            shared_replay_buffer (instance of Class): instance of Class ReplyBuffer, to be shared by by all separate instances of this class (ie agents)
        """

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.t_step = 0  # A counter that increases each time the "step" function is executed
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNetwork(state_size, action_size, USE_BATCH_NORM, random_seed,
                                        fc1_units=FC1_UNITS, fc2_units=FC2_UNITS, fc3_units=FC3_UNITS).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, USE_BATCH_NORM, random_seed,
                                         fc1_units=FC1_UNITS, fc2_units=FC2_UNITS, fc3_units=FC3_UNITS).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR,
                                          weight_decay=WEIGHT_DECAY_ACTOR)
        # self.actor_optimizer = optim.RMSprop(self.actor_local.parameters(), lr=LR_ACTOR,
        #                                      weight_decay=WEIGHT_DECAY_ACTOR)  # Untested

        # Critic Network (w/ Target Network)
        self.critic_local = CriticNetwork(state_size, action_size, USE_BATCH_NORM, random_seed,
                                          fc1_units=FC1_UNITS, fc2_units=FC2_UNITS, fc3_units=FC3_UNITS).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, USE_BATCH_NORM, random_seed,
                                           fc1_units=FC1_UNITS, fc2_units=FC2_UNITS, fc3_units=FC3_UNITS).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY_CRITIC)
        # self.critic_optimizer = optim.RMSprop(self.critic_local.parameters(), lr=LR_CRITIC,
        #                                       weight_decay=WEIGHT_DECAY_CRITIC)  # Untested

        # Make sure target is initiated with the same weight as the local network
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Setting default modes for the networks
        # Target networks do not need to train, so always eval()
        # Local networks, in training mode, unless altered in code - eg when acting.
        self.actor_local.train()
        self.actor_target.eval()
        self.critic_local.train()
        self.critic_target.eval()

        # Action Noise process (encouraging exploration during training)
        # Could consider parameter noise in future as a potentially better alternative / addition
        if ACTION_NOISE_METHOD == 'initial':
            self.noise = InitialOrnsteinUhlenbeckActionNoise(shape=(num_agents, action_size), random_seed=random_seed,
                                                             x0=0, mu=0,
                                                             theta=NOISE_THETA, sigma=NOISE_SIGMA)
        elif ACTION_NOISE_METHOD == 'adjusted':
            self.noise = AdjustedOrnsteinUhlenbeckActionNoise(shape=(num_agents, action_size), random_seed=random_seed,
                                                              x0=0, mu=0,
                                                              sigma=NOISE_SIGMA, theta=NOISE_THETA, dt=NOISE_DT,
                                                              sigma_delta=NOISE_SIGMA_DELTA, )
        else:
            raise ValueError('Unknown action noise method: ' + ACTION_NOISE_METHOD)

        # Replay buffer
        # If there is a shared instance of the ReplayBuffer Class, use it as this instance' replay_buffer,
        # else create a new instance of the ReplayBuffer Class (ie it's own replay_buffer separate from other instances
        if shared_replay_buffer is not None:
            self.replay_buffer = shared_replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE,
                                              batch_size=BATCH_SIZE,
                                              sampling_method=REPLAY_BUFFER_SAMPLING_METHOD,
                                              random_seed=random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        self.t_step += 1

        # Save experience / reward
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

        #Reduce optimizer learning rate over steps
        # This may further reduce instability because it reduces the chance of overshooting the target on the gradient
        if self.t_step % LR_DECAY_99_EVERY_ACTOR == 0:
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                              lr=LR_ACTOR * (0.99 ** (self.t_step // LR_DECAY_99_EVERY_ACTOR)),
                                              weight_decay=WEIGHT_DECAY_ACTOR)
        if self.t_step % LR_DECAY_99_EVERY_CRITIC == 0:
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                               lr=LR_CRITIC * (0.99 ** (self.t_step // LR_DECAY_99_EVERY_CRITIC)),
                                               weight_decay=WEIGHT_DECAY_CRITIC)

        # Learn, if enough samples are available in replay_buffer, every UPDATE_EVERY steps
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_action_noise=False):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()  # train state is set right before actual training
        with torch.no_grad():  # All calcs here with no_grad, but many examples didn't do this. Weirdly, this is slower..
            return np.clip(self.actor_local(states).cpu().data.numpy() + (self.noise.sample() if add_action_noise else 0), -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): reward discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        self.actor_local.train()  # critic_local is always in train state, but actor_local goes into eval with acting

        # Critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CLIP_GRADIENT_CRITIC:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if CLIP_GRADIENT_ACTOR:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # Soft-Update of Target Networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update target model parameters from local model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
