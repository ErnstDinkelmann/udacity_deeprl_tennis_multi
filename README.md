![trained agent](trained_agent.gif)



## Introduction
This project uses a multi-agent co-ordinator working with two agents using Deep Deterministic Policy Gradient (DDPG) to collaborate on playing tennis.. The environment is a pre-built variant of the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. This project is being done as part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).



## Environment
This environment is a pre-built version of the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) example one. It is therefore not necessary to install Unity itself.

There are two rackets that we control, on opposite sides of a net. The rackets can be controlled left and right with a single action variable and also made the "jump" with a second variable (this just raises that racket from the lowest default level relatively quickly upwards). The rackets are slightly tilted towards the net, making it feasible to hit the ball back over the net without controlling the angle of the racket.

A ball is observed which adheres to basic physics. The purpose is to keep the ball in play. The same racket can hit the ball more than once and this is regarded valid - ie the balls stays in play. The game stops as soon as the ball hits the grounds or any out-of-bounds area at the back of the rackets.

Rewards are given when the ball crosses the net.

It's an episodic task with a maximum of 1001 timesteps per episode.

#### States
The state space has 24 dimensions per agent corresponding to the position and velocity of the ball and racket. Given this information, the agent has to learn how to best select actions.

Note these are separate agents and the observations are from the perspective of the agent itself. Therefore, although both agents observe the same ball, it will be observed from their own perspective and not a single view (or world view).

#### Actions
Two continuous simultaneous actions are available, corresponding to: 1) moving towards or away from the net and 2) "jumping". Each action value is standardized and has to be in the rage [-1, 1]. Values outside of this range is clipped by the environment but it is preferred to do the clipping before sending to the environment.

#### Rewards
A reward of +0.1 is given when the ball crosses the net and -0.01 if the ball hits the ground or out-of-bounds areas.

#### Goal
Broadly the goal is to keep the ball in play and hit it back-and-forth from one racket to the next and thereby maximize the reward.

The task is episodic, and in order to solve the environment, we must get an average score of +0.5 per over 100 consecutive episodes. The score per episode is the one scored by the agent with the largest score. An agent's score is just the sum of it's rewards over all timesteps observed in an episode. 



## Installation

#### Operating system and context
The code was only run in Ubuntu 18.04. It may be possible to get it working on other operating system, but this is untested.

The project came with Jupyter Notebook files, but I've decided not to use these are I am more comfortable with PyCharm and have therefore decided to do the project in PyCharm. Therefore, all files with python code will be `.py` files.

The recommended method to work with the project is to run the main.py from the command line (in a terminal) with the intended parameters. More details on this below.

#### Pre-requisites
Make sure you having a working version of [Miniconda](https://conda.io/miniconda.html) (the one I used) or [Anaconda](https://www.anaconda.com/download/) (should also work, but untested) on your system.

#### Step 1: Clone the repo
Clone this repo using `git clone https://github.com/ErnstDinkelmann/udacity_deeprl_tennis_multi.git`.

#### Step 2: Install Dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

Linux:
```
conda create --name drlnd python=3.6
source activate drlnd
conda install -y pytorch -c pytorch
pip install unityagents
```

#### Step 3: Download Tennis environment
You will also need to install the pre-built Unity environment, but you will NOT need to install Unity itself (this is really important, as setting up Unity itself is a lot more involved than what is required for this project).

Select the appropriate compressed file for your operating system from the choices below:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Download the file into the top level directory of this repo and extract it. The code as it is, assumes that the environment file (`Tennis.x86` for 32-bit systems or `Tennis.x86_64` for 64-bit systems) is located with the `/Tennis_Env/` directory in the root of the repo.



## Files in the repo
This is a short description of the files in the repo (that comes with the repo) or may be generated as output by the code when running:

* `main.py`: the main file containing high-level training function, a function for viewing a trained agent, as well as a function for plotting the results. Note that the directory location of this file is detected and used as a parameter in the code. If you are not executing from the command line, you may need to make adjustments to the code. This is the file that will be run from the command line as follows:

    * There is one command-line arguments:
        * `--mode` with possible values `train` for training an agent (the default), `view` for viewing a trained agent and `plot` for plotting the results after training.
    * E.g. to train the agent: `python main.py --mode train` or, since training is the default behaviour, `python main.py` will do the same thing.
    * E.g. to view a trained agent: `python main.py --mode view`. You will need a `actor_checkpoint_0.pth` file (for the first agent/racket) and a `actor_checkpoint_1.pth` file (fort he second agent/racket) in the root of the repo for this to work.

* `parameters.py`: the file with all the hyper-parameters that we chose to expose (essentially the ones we felt were important during some stage of the development and understanding). Actual parameters are merely global python constants.

* `agent.py`: contains the MultiDdpgAgent class, which co-ordinates the two agents (rackets) being instances of the DdpgAgent class (also contained in the file). The DdpgAgent class controls an agent interaction and learning with/from the environment. Each agent learns the mapping of states -> actions with the help of a critic that learns states + actions -> action value. These mappings are learnt as neural networks specified in the networks.py file.

* `networks.py`: contains the neural networks set up in pytorch. There are two classes specified: the ActorNetwork and the CriticNetwork, both of which are used by the separate agents for learning.

* `replay_buffer.py`: contains the ReplayBuffer class, which serves as the memory of the agents. It is a fixed length list that stores experiences of the agent within the environment. The agent then samples from these past experiences to learn from, as opposed to learning directly from the environment as it's experienced. A single replay buffer could either be shared between the agents (speeding up learning through shared experience), or each agent can have their own experience buffer (making the agents completely independent). We chose the former as a form of "collaborating".

* `noise.py`: contains two classes which generates noise, only one of which is used, chosen by a parameter. They work differently and details are within the file. The noise is added to the actions so that we explore the environment.

* `actor_checkpoint_x.pth` and `critic_checkpoint_x.pth`: contains the saved weights of the specified neural network, for an already trained agent right after they achieved the requisite performance to "solve" the environment. The `x` in the filename is the agent number. There are also `_eps` versions where the eps is the episode at which the save was done - at episode 100, 200, 300, etc.



## Train the agent
Refer to the `main.py` file description above for help on how to start training.

This will start the Unity environment and output live training statistics to the command line. Even after "solving" the environment, it will continue to save checkpoint files with the weights of the networks at episode 100, 200, 300, etc thereafter. You may interrupt the code at any time.

Feel free to experiment with modifying the hyper-parameters to see how it affects training.

You do not need an extremely fast GPU (CUDA) for this purpose and can even completely the training on a CPU if you are willing to wait a little.


## View the trained agent perform
Refer to the `main.py` file description above for help on how to start viewing.
This will load the saved weights from checkpoint files (`actor_checkpoint_0.pth` and `actor_checkpoint_1.pth`).  The weights of two previously trained agents (rackets) is already included in this repo.



## Report
See the [report](Report.md) for more insight on the technical modelling aspects of the project and how we arrived at a solution.
