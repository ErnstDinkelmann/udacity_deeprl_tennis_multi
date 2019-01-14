This report details the methodology used in evaluating and improving agent learning performance. It provides more insight on the technical modelling aspects of the project and how we arrived at a solution.



## Introduction
This project uses a multi-agent co-ordinator working with two agents using Deep Deterministic Policy Gradient (DDPG) to collaborate on playing tennis. The environment is a pre-built variant of the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. This project is being done as part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).



## Algorithm
This algorithm implements a multi-agent approach to solving the problem. This will be explained in more detail below. But it's important to note that this problem could have been solved with almost exactly the same algorithm that was used for the reacher-continuous problem. We also observe multiple separate agents simultaneously. However, in that environment, even though multiple agents are observed, only a single agent instance is learning to act. It's learnings were then applied to the observations that the different agents experienced.

For the tennis environment here it is still possible to do it in this way as the observations are from the perspective of agent's themselves and except for facing and playing in opposite directions, they are identical. And even the directions have been accounted for in the state-space (observations). For example, to be practical, if each racket/agent was next to the net at the lowest point, with no actions, and a different ball travelled towards each agent with the same trajectory, they would have exactly the same state observations.

However, the project requires the use of a "true" multi-agent approach, where the agents use a collaborative approach to solving the environment challenge. Hence, this is what is implemented. A MultiDdpgAgent Class was created within the `agent.py` file, which we regard as a co-ordinator of individual agents (2 in the case of the two rackets, but they could easily be multiple). The co-ordinator does not learn itself (it does not have a neural network), it does not act, it does not observe. It is merely a conduit - the observations are received by it and it passes them on to the individual agents in an appropriate way. The agents in turn store the experience in a reply buffer (perhaps shared between the agents) and learns from the experience from time to time. when needed the co-ordinator asks the agents for actions which it receives and passes these back to the environment.

It is certainly possible to do what the co-ordinator class does, without a separate class, but it makes it a lot simpler if there is a conduit layer.

As mentioned, the project requires a collaborative approach. So the individual agents should "collaborate" in some way. If they are completely separate, learning from their own experience only and not transferring any learnings, nor sharing reward in the loss functions, they are not collaborating. Merely evaluating the total score as something that both agents contribute to does not imply that the individuals are collaborating in any way. Think of a soccer team which wins or loses as a team but that does not force the players to collaborate. We thought there should be something more explicit. Hence we've used a shared replay buffer - in other words, the experiences of both agents are stored in the same replay buffer, accessible by both agents when learning. Other approaches are perhaps possible such as using some kind of shared rewards in the loss function.


This individual agents each implements the Deep Deterministic Policy Gradient (DDPG) algorithm, also within the `agent.py` file.

Details about DDPG:

* It is a mainly policy gradient (PG) method, as opposed to a values based method. Therefore, it attempts to learn a function to map environment states to optimal actions directly.
* PG methods, such as DDPG, allow for multiple continuous actions within the action space, which could be taken simultaneously at a timestep. This is especially useful in our case where the actions are 2 continuous variables, each of which must have a value in the range [-1, 1] at each timestep. It would have been challenging to implement value based methods to this problem (e.g. some forms of discretization of the action space).
* DDPG specifically was originally introduced as an actor-critic method. Actor-Critic (in general) means that there is one network that learns a policy (the actor) and another network that learns a value function (the critic). The critic's output (in general) forms a 'baseline' for the actor, which reduced variance without increasing bias and therefore speeds up the learning process.
* Some critics see DDPG as an approximate DQN (Deep Q Network) usable for continuous action spaces, because the critic in this case does not represent a learned baseline, but rather approximate the maximizer over the Q-values over the next state.
* Off-policy training - The agent learns from the experience of another agent with a different (e.g. more exploratory) policy, In fact, we make use of explicit noise added to the action to encourage exploration, but the trained agent will not have this noise added when testing.
* Bootstrapping - update estimates of the values of states based on estimates of the values of successor states. Our replay buffer stores a tuple of (state, action, reward, next_state and done_status), which is the only bit of information needed to enable learning - hence bootstrapped.
* Function approximation - through neural networks (specified in the next section), learns a function that maps states to actions even for unseen states in the case of the actor or the value of the next state in the case of the critic.

PG methods have divergence issues where the agent does not converge on a policy but instead oscillates all over the place, or improve only to crash, or never learns due to sensitivity to hyper-parameters.

To mitigate this, we employed a couple of modifications, including:

* Experience Replay - a finite memory buffer of past experiences that the agent can sample from during learning, and
* Fixed (or rather slowly changing) targets - use a second neural network with weights that do not change as quickly as the online network and is used when calculating the targets.
* Gradient Clipping - limit the gradient during back-propagation in the networks, to avoid too extreme adjustments to weights, which may potentially cause instability.
* Slowly reducing action noise - reduce the noise and hence exploration as training progresses, which leads to more stability during later episodes.
* Batch normalisation - normalise each batch of inputs to each layer in the networks. Eventually we turned this off, which actually improved our convergence properties.

All of these help stabilize convergence but do not guarantee it.



## Neural networks model architectures
The neural networks are implemented in the 'networks.py' file, using the PyTorch framework.

As mentioned, there are two networks: an actor and a critic. The basic idea was fairly deep networks with fully connected layers. We tried a shallow network and many different units and therefore went deeper and more units eventually. The weights are reset to sit between a small negative and a small positive number, relative to the number of hidden units. The LeakyRelu Activation function (where the negative side is not set completely to zero but to a small negative number relative to the incoming number), showed better performance than the standard Relu Activation function.

The architectures we ended up with are described below.

#### Actor Network
The actor network can be illustrated as follows:

* States (24 units), into
* LeakyRelu Activation, into
* Fully Connected Layer (1024 units), into
* LeakyRelu Activation, into
* Fully Connected Layer (1024 units), into
* LeakyRelu Activation, into
* Fully Connected Layer (1024 units), into
* Tanh Activation, into
* Actions (2 units)

#### Critic Network
The critic network can be illustrated as follows:

* States (24 units), into
* LeakyRelu Activation, into
* Concatination with Actions (2 units), all into
* Fully Connected Layer (1024 units), into
* LeakyRelu Activation, into
* Fully Connected Layer (1024 units), into
* LeakyRelu Activation, into
* Fully Connected Layer (1024 units), into
* Action-Value (1 unit)



## Parameters

#### General parameters
* RANDOM_SEED = 0 (Global random seed ensuring reproducible results)
* MAX_N_EPISODES = 10000 (Overall max number of training episodes. Practically we stop earlier)
* MAX_N_STEPS = 1001 (Maximum number of training steps per episode before starting a new episode)
* WIN_SCORE = 0.5 (Project goal. Once reached, will save networks' weights to file)
* WIN_QUEUE_LEN = 100 (Number of episodes over which the total reward is averaged to get score)

#### Agent parameters
* REPLAY_BUFFER_SIZE = int(1e5) (replay buffer size)
* REPLAY_BUFFER_SAMPLING_METHOD = 'uniform' (one of 'uniform'/'geometric'/'linear'. refer replay_buffer.py)
* BATCH_SIZE = 1024 (mini-batch size)
* GAMMA = 0.9999 (discount factor for rewards)
* TAU = 1e-3 (for soft update of target parameters)
* LR_ACTOR = 2e-5 (learning rate of the actor)
* LR_CRITIC = 2e-4 (learning rate of the critic)
* LR_DECAY_99_EVERY_ACTOR = 1e10 (Optimizer re-declared with LR *= 0.99 every this number of steps)
* LR_DECAY_99_EVERY_CRITIC = 1e10 (Optimizer re-declared with LR *= 0.99 every this number of steps)
* WEIGHT_DECAY_ACTOR = 0 (L2 weight decay actor)
* WEIGHT_DECAY_CRITIC = 0 (L2 weight decay critic)
* USE_BATCH_NORM = False (whether to use batch normalisation)
* UPDATE_EVERY = 1 (Update weights this often in steps)
* CLIP_GRADIENT_ACTOR = True (Whether to clip the Actor Network's Parameter's Gradient to a norm of 1)
* CLIP_GRADIENT_CRITIC = True (Whether to clip the Critic Network's Parameter's Gradient to a norm of 1)

#### Parameters for noise added to actions
Action noise is broadly based on a Ornstein-Uhlenbeck process.

* ADD_ACTION_NOISE_TRAINING = True (Whether to add noise to the actions in training)
* ACTION_NOISE_METHOD = 'adjusted' (one of 'initial'/'adjusted'. refer noise.py)
* NOISE_SIGMA = 1 (initial variability)
* NOISE_THETA = 1 (extent of continuing in a direction (like a momentum of sorts))
* NOISE_DT = 1 (extent of short-term variability)
* NOISE_SIGMA_DELTA = 0.9999 (overall variance reduction factor)

#### Network layer parameters
* FC1_UNITS = 1024 (Number of fully connected units in the 1st hidden layer, both actor and critic)
* FC2_UNITS = 1024 (Number of fully connected units in the 2nd hidden layer, both actor and critic)
* FC3_UNITS = 1024 (Number of fully connected units in the 3rd hidden layer, both actor and critic)

Many of these parameters could potentially be adjusted for improved performance.



## Results and thoughts
The graph below shows the total maximum score per episode.

![training_score_by_episode](training_score_by_episode.png)

The graph below shows the average total maximum score, over the previous 100 episodes, as the episodes progressed.

![training_score_100_episodes](training_score_100_episodes.png)

The winning average score of +0.5 over the previous 100 episodes is achieved after episode 864.
In other words, the average of the 100 episodes before the 865th episode tops the required winning score of +0.5. 

We included two sets of checkpoint files (i.e. two for each of the two agents): 

* `actor_checkpoint_0.pth` and `actor_checkpoint_1`: the weights of first and second agent, the moment the goal is reached.
* `actor_checkpoint_0_1000.pth` and `actor_checkpoint_1_1000`: the weights of first and second agent, at episode 1000, which is close to it's maximum training performance.

In testing (viewing a trained agent), using the `_1000.pth` checkpoints, an average score of 0.84 is achieved over the first 50 episodes.



#### Caveats, thoughts and things to consider

* A lot of improvement is likely possible given a few adustments to hyper-parameters. We've not extensively done optimisation.
* As seen on the graph, the score stays flat for the first 650 odd episodes, while the agents experience more and more and hence learns. From this point it seems to come together. There is still a lot of variability in the performance from that point forward, but this is likely due to the extent of action noise that is added to the actions. However, performance then slowly starts dropping off. This is obviously unwanted. It's fine to stabilise at a peak performance with slights ups and downs from that, but for it to continue dropping like this is undesirable. This behaviour is experienced even with all the stabilising adjustments made to the model as mentioned above. What I imagine we need in addition are learning rates for the networks that decay as well over episodes/timesteps. Parameters were created for this, but not tested as yet.
* Having said that, looking at the moving average graph (over 100 episodes), those drops in performance could just be a path towards improved performance in the end. One can see some swings downwards and upwards. It could be that the agents are now experiencing states for the first time (due to action seed), where it is not performing well, but in the end this is good - they have to experience these states that may occur in a test/real game. We did not run the training longer to see how this turns out eventually.
* The training performance is very sensitive to hyper-parameters and the seed. To the extent that it almost feels like the hyper-parameter combination here is one of a few that solves for the given seed value (taking into account the setup and architecture, not to say DDPG in general). There are many examples, but here are but a few.
    * E.g. for some seed values, the agent takes a lot longer to make progress in training.
    * E.g. making minor adjustments to number of units in the fully connected layers, leads to completely different performance, which does not feel intuitive.
    * E.g using slightly different noise parameters also has a large unintuitive effect.
* Reproducible results are great, but when one will automate the hyper-parameter tuning in something like Ray (met previously, never used), one will have to work with a larger set of sets and average over them. This may be applicable to real world problems as well as it's not only simulations that have seeds. On our agent's side there are initialisations that depend on seeds.
* The action noise is an adjusted Ornstein-Uhlenbeck process, which reduces variance as the timesteps progress. We thought this worked well and a lot better than some of the implementations we've seen. When looking at the actual noise generated compared to other implementations, it just feels a lot more intuitive as well, which we thing is important.
* RMPSProp (As an optimizer), instead of Adam, also solves challenge, but takes a lot longer.
* We attempted many different neural network setups. In the end, a deeper net with more units and a slower learning rate seemed to work better. However, many people reported getting good results with simpler networks. We were not able to reproduce these results. This is a bit worrying, but again, the full combination of hyper-parameters determines things like this and ours was probably just slightly different anyway.
* Within the networks, we also attempted dropout layers to try to improve stability and avoid seed dependence, but this did not work better than without it.
* We also attempted initialising the layer biases manually (similar to weights), but could not get better results than the automatic initialisation.
* Many people reported better results when using batch normalisation on each layer within the network. This did not work for our setup. It is unclear why. If this can be understood and solved, it could improve the stability in later episodes perhaps.
* The size of the replay buffer is a method whereby we control the length of the "memory" of the agent. The smaller, the more the learning is focused on recent experience and in fact if learning continues for long enough, the initial experience will have very little influence (tending to zero). And the opposite for larger buffers. However, there must be a more explicit way to allow for this effect and the solution probably lies in keeping more experience, but sampling more intelligently. More about this in the "Potential Improvements" section below.
* Gradient clipping, does improve our training. We are not sure what the best clipping value should be and perhaps this could a hyper-parameter as well.
* It has been alluded to, but careful thought is needed as to what extent you allow the agents/rackets to "collaborate" and what it might mean. We could for example have a shared critic network - evaluating values of next states. This could considerably improve the performance as it effectively halves the number of weights that needs to be trained for the critic. The point is though, "collaboration" is open to interpretation in our opinion.



## Potential Improvements
There are plenty more things which could be tried to further improve performance, such as:

* Systematic hyper-parameter optimisation, using an optimised library such as Ray. Importantly, as mentioned above, different seeds should be used and averaged over to improve robustness in a real world scenario.
* Testing other reply buffer sampling methods that are not just uniform, reducing the sensitive to the size of the buffer and improving performance. The context was set the one of the last points in the section above, but the idea is to explicitly sample more often from more recent experience. There could be many different ways. We did not go into detail on this, but attempted two implementations, which can be references in the `replay_buffer.py` file. So far the limitation was that it sampled a lot slower than the uniform and for no reason other than that the libraries used was not optimised to sample batches from different distributions.
* Instead of adding noise to the actions, it has been reported that adding noise to the network parameters (weights) in the forward pass of the network, is an improved way of adding exploration.
* Use prioritised replay. We've not seen an implementation of this, but it could be something to look into.
* Completely different models (e.g. PPO, D4PG, etc) could be attempted.