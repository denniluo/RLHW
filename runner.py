# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:34:50 2018

@author: Maria Dimakopoulou
"""

from environments import CartPole
from agents import ConstantAgent, RandomAgent, SARSA, EpisodicQLearning
from agents import TabularFeatures

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1704)


horizon = 100
episode_count = 10000
verbose = False

environment = CartPole(verbose=verbose)
result_qlearn_e_greedy = []

experiment_counter = 0
for eps in np.arange(0.01, 0.22, 0.2/10):
  for gam in np.arange(0.01, 0.12, 0.1/5):
    agent = EpisodicQLearning(num_action=len(environment.action_space),
                        feature_extractor=TabularFeatures(5, 5, 11, 11),
                        gamma=gam, epsilon=eps)


    reward_per_episode = np.zeros(episode_count)

    for episode in range(episode_count):
      # Initialize state and time period.
      current_state = environment.reset()
      # Pick the action
      current_action = agent.pick_action(current_state)
      time = 0

      # Run the episode.
      while True:
        # Collect the reward, next state and continue probability.
        step = environment.step(current_action)
        reward = step.reward
        next_state = step.new_obs
        p_continue = step.p_continue
        # Pick the next action.
        next_action = agent.pick_action(next_state)

        reward_per_episode[episode] += reward
        # Update the agent.
        agent.update_observation(obs=current_state, action=current_action,
                                 reward=reward, new_obs=next_state,
                                 p_continue=p_continue, new_action=next_action)
        # Update the state and the time period.
        current_state = next_state
        current_action = next_action
        time += 1
        # Continue or stop the episode.
        terminate = np.random.random() > p_continue or time >= horizon
        if terminate:
          break

    print("**********************************************")
    print("epsilon = {}, gamma = {}".format(eps, gam))
    print("last 100 episode mean total reward is {}".format(np.mean(reward_per_episode[-1])))
    result_qlearn_e_greedy.append([eps, gam, np.mean(reward_per_episode[-100:])])
    handles = []
    plt.figure(experiment_counter)
    experiment_counter += 1
    h, = plt.plot(range(episode_count), reward_per_episode, label=str(agent))
    handles.append(h)
    plt.xlabel("Episode (Q learning with e-greedy and epsilon = {}, gamma = {})".format(eps, gam))
    plt.ylabel("Reward")
    plt.ylim([0, None])
    plt.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, -0.25))
    plt.savefig("qlearning_epsilon{}_gamma{}.png".format(eps, gam))
    plt.close()

print("print result for Q learning + e greedy")
print(result_qlearn_e_greedy)
