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

verbose = False
render = True

horizon = 100
episode_count = 10000

environment = CartPole(verbose=verbose)
eps = 0.1
gam = 0.1
agent = EpisodicQLearning(num_action=len(environment.action_space),
                    feature_extractor=TabularFeatures(5, 5, 11, 11),
                    epsilon=eps, gamma=gam)


state_action_list_per_episode = [[] for episode in range(episode_count)]
reward_per_episode = np.zeros(episode_count)

for episode in range(episode_count):
  if verbose:
    print("********** EPISODE l={} START *************".format(episode))

  # Initialize state and time period.
  current_state = environment.reset()
  # Pick the action
  current_action = agent.pick_action(current_state)
  time = 0

  # Run the episode.
  while True:
    if verbose:
      print("t={}".format(time))
    # Collect the reward, next state and continue probability.
    step = environment.step(current_action)
    reward = step.reward
    next_state = step.new_obs
    p_continue = step.p_continue
    # Pick the next action.
    next_action = agent.pick_action(next_state)
    # Log for visualization.
    state_action_list_per_episode[episode].append((current_state,
                                                   current_action))
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
    if terminate and verbose:
      print("********** EPISODE l={} END *************\n" \
          "episode_reward={}, cumulative_reward={}\n"
          .format(episode, reward_per_episode[episode],
                  np.cumsum(reward_per_episode)[episode]))
    if terminate:
      break


# Visualize the execution of some of the episodes.
if render:
  episodes_to_render = [100, 500, 1000, 5000, 10000]
  for episode in episodes_to_render:
    episode -= 1
    print("********** RENDERING EPISODE l={} *************\n" \
          "episode_reward={}, cumulative_reward={}\n"
          .format(episode, reward_per_episode[episode],
                  np.cumsum(reward_per_episode)[episode]))
    environment.render(state_action_list_per_episode[episode])


# Plot the reward per episode.
handles = []
h, = plt.plot(range(episode_count), reward_per_episode, label=str(agent))
handles.append(h)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.ylim([0, None])
plt.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, -0.25))
plt.show()