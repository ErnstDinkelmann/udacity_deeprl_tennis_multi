# coding: utf-8
# Author: Ernst Dinkelmann

# We begin by importing some necessary packages.  If the code cell below returns an error,
# double-check that you have installed:
# [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and
# [NumPy](http://www.numpy.org/).

# Also confirm that you have activated your environment with the necessary packages, ie using the
# intended python interpreter

from unityagents import UnityEnvironment  # The environment
import numpy as np
from parameters import *
from agent import MultiDdpgAgent
from collections import deque
import matplotlib.pyplot as plt
import torch
import os
import inspect
from argparse import ArgumentParser
import time


def train(n_episodes=MAX_N_EPISODES, max_n_steps=MAX_N_STEPS, win_queue_len=WIN_QUEUE_LEN, win_score=WIN_SCORE,
          output_folder='../'):
    """
    Training function for our agent, which will let the agent choose actions and get responses from the environment.
    This is done in iteration over episodes and steps within episodes.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_n_steps (int): maximum number of steps per episode
        win_queue_len (int): number of episodes over which the scores are averages to test against the win condition.
        win_score (int): the average score over win_queue_len episodes, which would be regarded as a win condition.
        output_folder (str): full path to where the model checkpoints (ie weights), as well as other metrics
            should be saved. There may be multiple checkpoint files, depending on the model involved.
    """

    assert os.path.exists(output_folder), 'Checkpoint folder does not exist'

    score_tot_all_eps = []  # list containing scores from each episode
    score_tot_queue_eps = deque(maxlen=win_queue_len)  # fixed length queue over which we will average to test win
    num_saves = 0
    t_step = 0  # a counter for total number of steps cumulatively

    for i_episode in range(1, n_episodes + 1):
        timeStart = time.time()
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations.reshape([num_agents, num_obs])  # get the current state
        scores_one_eps = np.zeros(num_agents)  # initialize the scores for each agent
        last_t = max_n_steps
        # agent.reset()  # not doing this each episode as we actually want our noise to reduce over time and not reset.

        for t in range(max_n_steps):
            actions = agent.act(states, add_action_noise=ADD_ACTION_NOISE_TRAINING)  # agent returns action(s), based on state
            # print('Left: Horz: {:.2f} Jump: {:.2f}          Right: Horz: {:.2f} Jump: {:.2f}'.format(actions[1, 0], actions[1, 1], actions[0, 0], actions[0, 1]))
            # Agent 0: is right, Action 0 (horz): <0 is away from net, Action 1 (jump): >= 0.5 is jump
            env_info = env.step(actions)[brain_name]  # send the action(s) to the environment
            next_states = env_info.vector_observations.reshape([num_agents, num_obs])  # get the next states, given as array: shape (#agents,#obs)
            rewards = np.array(env_info.rewards)  # get the rewards, convert to array for consistency
            dones = np.array(env_info.local_done)  # episode end indicator(s), convert to array for consistency
            agent.step(states, actions, rewards, next_states, dones)  # records experience and learns potentially
            states = next_states
            scores_one_eps += rewards  # ie keeping the total score for each agent separate during the episode
            if dones.any():  # this specific logical could differ from environment to environment
                last_t = t + 1
                break

        t_step += last_t

        score_tot_queue_eps.append(np.max(scores_one_eps))  # save most recent score, which is the total score of agent which had the highest total score
        score_tot_all_eps.append(np.max(scores_one_eps))  # save most recent score
        timeDuration = time.time() - timeStart  # calc the time it took to complete the main steps in the episode
        print('\rEpisode {}\tNum steps: {}\tTot Num steps: {}\tDuration: {:.1f}'
              '\tShared Replay Buffer Size: {}\tEpisode Score: {:.2f}\t100_Episode Score: {:.2f}'
              .format(i_episode, last_t, t_step, timeDuration, len(agent.shared_replay_buffer),
                      np.max(scores_one_eps), np.mean(score_tot_queue_eps)))

        # output the scores_episode to a physical text file every 10 episodes (just in-case a battery dies or something)
        if (i_episode + 1) % 10 == 0:
            with open(strOutputFolder + '/scores_episode.txt', 'w') as f:
                for item in score_tot_all_eps:
                    f.write("%s\n" % item)

        # output the weights of the model(s) once win condition met and at episode 100,200, 300,... thereafter
        if np.mean(score_tot_queue_eps) >= win_score:
            if num_saves == 0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_tot_queue_eps)))
                print('Training will continue and the checkpoint will be overwritten at episode 100, 200, 300, ...')
                print('Saving a checkpoint now, you may interrupt code execution with eg Ctrl+C')
                for idx in range(num_agents):
                    torch.save(agent.ddpg_agents[idx].actor_local.state_dict(),
                               output_folder + '/actor_checkpoint_' + str(idx) + '.pth')
                    torch.save(agent.ddpg_agents[idx].critic_local.state_dict(),
                               output_folder + '/critic_checkpoint_' + str(idx) + '.pth')
                print('Saving done')
            else:
                if i_episode % 100 == 0:
                    print('\nSaving another checkpoint now, you may interrupt code execution with eg Ctrl+C')
                    for idx in range(num_agents):
                        torch.save(agent.ddpg_agents[idx].actor_local.state_dict(),
                                   output_folder + '/actor_checkpoint_' + str(idx) + '_' + str(i_episode) + '.pth')
                        torch.save(agent.ddpg_agents[idx].critic_local.state_dict(),
                                   output_folder + '/critic_checkpoint_' + str(idx) + '_' + str(i_episode) + '.pth')
            num_saves += 1

    env.close()

    return score_tot_all_eps

def view(n_episodes=1, output_folder='../'):
    """
    Viewing function for our agent, which will let the agent choose actions based on the training it previously did.
    This is done in iteration over a single episode, with graphics to see the agent perform.
    Loads a saved 'actor_checkpoint.pth' file to view a trained agent perform.
    Although we also saved the 'critic_checkpoint.pth' during training, it's not needed during viewing.

    Params
    ======
        n_episodes (int): the number of episodes the agents must be viewed with graphics
        output_folder (str): full path to the saved model weights, checkpoint files
    """

    # Lead saved weights
    for idx in range(num_agents):
        agent.ddpg_agents[idx].actor_local.load_state_dict(
            torch.load(output_folder + '/actor_checkpoint_' + str(idx) + '.pth'))
        # agent.ddpg_agents[idx].actor_local.load_state_dict(
        #     torch.load(output_folder + '/actor_checkpoint_' + str(idx) + '_1000.pth'))  # close to best performance

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations.reshape([num_agents, num_obs])  # get the current state
        scores_one_eps = np.zeros(num_agents)  # initialize the scores for each agent

        while True:
            actions = agent.act(states, add_action_noise=False)  # agent returns action(s), based on state
            # actions = np.clip(np.random.randn(num_agents, brain.vector_action_space_size))  # random normal actions
            env_info = env.step(actions)[brain_name]  # send the action(s) to the environment
            next_states = env_info.vector_observations.reshape([num_agents, num_obs])  # get the next states, given as array: shape (#agents,#obs)
            rewards = np.array(env_info.rewards)  # get the rewards, convert to array for consistency
            dones = np.array(env_info.local_done)  # episode end indicator(s), convert to array for consistency
            states = next_states
            scores_one_eps += rewards    # ie keeping the total score for each agent separate during the episode
            if dones.any():  # this specific logical could differ from environment to environment, works here
                break

        print("Episode {} Score: {:.2f}".format(i_episode, np.max(scores_one_eps)))  # total score of agent which had the highest total score
    env.close()

def plot_last_scores(output_folder='../'):
    """
    Function to plot the score by episode from a txt file.
    Also saves a png of the result.

    Params
    ======
    output_folder (str): full path to the scores_episode.txt folder.
    """

    with open(output_folder + '/scores_episode.txt', 'r') as f:
        lstScores = [float(x.strip()) for x in f.readlines()]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(len(lstScores)), lstScores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(output_folder + '/training_score_by_episode.png')
    plt.show()



if __name__ == '__main__':
    # Set up argument parsing for execution from the command line
    # There is only a single argument --mode, which can be either 'train' or 'view'
    # If running in console, comment out the parser lines and uncomment the manual setting of the args dictionary
    parser = ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='train (default), view, plot', metavar='MODE', default='train')
    parser.add_argument('--env', dest='env', help='single, multi (default)', metavar='ENV', default='multi')
    args = parser.parse_args()
    # args = parser.parse_args('--mode view'.split())

    # Automatic detection of the directory within which this py file is located will only work if the py file is
    # executed from command line. When executed within a console within an IDE for example, the context is different
    # In that case, just manually specify the strHomeDir as the full absolute directory within which this py file
    # is located.
    strHomeDir = str(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) + '/'
    # strHomeDir = '/home/ernst/Projects/udacity_deeprl_tennis_multi/'

    # We also set the full path to the Tennis Environment file - either the single or the multi agent versions
    # By default it assumes the applicable Tennis environment files lies either within a directory 'Tennis_Env'
    # which is located in the directory of the current py file.
    # You may download the environment as per instructions in the readme file
    strEnvFile = strHomeDir + 'Tennis_Env/Tennis.x86_64'

    # We also set the directory to save to and read from the saved network weights.
    # By default we save/read in strHomeDir
    # If the folder specified does not exist, it is created.
    strOutputFolder = strHomeDir
    if not os.path.exists(strOutputFolder):
        os.makedirs(strOutputFolder)

    # Change the working directory to the above directory - this just helps for potential relative path imports
    # Again, this will not necessarily work in the case of console execution as a different context may apply
    os.chdir(strHomeDir)

    if args.mode == "train":
        print('--mode train initiated')

        # Print parameters for this training run from the parameters.py file
        # Just helped me connect results with parameters if I made changes to things mid-runs.
        num_blanks = 0
        with open(strHomeDir + 'parameters.py', 'r') as f:
            for line in f.readlines():
                if len(line) == 0:
                    num_blanks += num_blanks
                    if num_blanks == 2:
                        break
                else:
                    num_blanks = 0
                    print(line)

        # Start the environment, with no_graphics=True to avoid visual elements, to speed up training.
        env = UnityEnvironment(file_name=strEnvFile, no_graphics=True, seed=RANDOM_SEED)

        # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
        # Here we check for the first brain available, and set it as the default.
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # Determine number of agents and number of observations per agent
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        num_agents = len(env_info.agents)
        num_obs = env_info.vector_observations.shape[1]
        print('Number of Agents:', num_agents)
        print('Number of Obs:', num_obs)

        # Instantiate our agent with the implied state and action space sizes as well as the number of agents
        agent = MultiDdpgAgent(num_agents=num_agents,
                               state_size=num_obs,
                               action_size=brain.vector_action_space_size,
                               random_seed=RANDOM_SEED)

        scores = train(output_folder=strOutputFolder)  # ...can do something with the scores here if needed.
    elif args.mode == 'view':
        print('--mode view initiated')

        # Start the environment, with no_graphics=False to see visual elements.
        env = UnityEnvironment(file_name=strEnvFile, no_graphics=True, seed=RANDOM_SEED)

        # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
        # Here we check for the first brain available, and set it as the default.
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # Determine number of simultaneous agents and number of observations per agent
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        num_agents = len(env_info.agents)
        num_obs = env_info.vector_observations.shape[1]
        print('Number of Agents:', num_agents)
        print('Number of Obs:', num_obs)

        # Instantiate our agent with the implied state and action space sizes as well as the number of agents
        agent = MultiDdpgAgent(num_agents=num_agents,
                               state_size=num_obs,
                               action_size=brain.vector_action_space_size,
                               random_seed=RANDOM_SEED)

        view(n_episodes=50, output_folder=strOutputFolder)
    elif args.mode == 'plot':
        print('--mode plot initiated')
        plot_last_scores(output_folder=strOutputFolder)
    else:
        raise ValueError('Unknown --mode parameter: ' + args.mode)
