import os
from datetime import datetime, timedelta
import argparse
import itertools
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque, namedtuple

from agent import Agent
     
import gymnasium as gym

# if __name__ == '__main__':
#     # Parse command line inputs
#     parser = argparse.ArgumentParser(description='Train or test model.')
#     parser.add_argument('hyperparameters', help='')
#     parser.add_argument('--train', help='Training mode', action='store_true')
#     args = parser.parse_args()

# env = gym.make("LunarLander-v3", render_mode="rgb_array") # discrete by default
# state_shape=env.observation_space.shape
# state_size=env.observation_space.shape[0]
# number_action=env.action_space.n
# print('State Shape :',state_shape)
# print('State Size :',state_size)
# print('Number of Actions :',number_action)
     
     
# #learning_rate=5e-4
# #minibatch_size=200
# #discount_factor=0.99
# #replay_buffer_size=int(1e5)
# #interpolation_parameter=1e-3


# agent = Agent(state_size,number_action, hyperparameter_set=args.hyperparameters)

# number_episodes=2000
# maximum_number_timesteps_per_episode=1000
# epsilon_starting_value=1.0
# epsilon_ending_value=0.01
# epsilon_decay_value=0.995
# epsilon=epsilon_starting_value
# scores_on_100_episodes=deque(maxlen=100)

# for episode in range(1,number_episodes+1):
#   state,_=env.reset()
#   score=0
#   for t in range(maximum_number_timesteps_per_episode):
#     action=agent.act(state,epsilon)
#     next_state,reward,done,_,_=env.step(action)
#     agent.step(state,action,reward,next_state,done)
#     state=next_state
#     score+=reward
#     if done:
#       break
#   scores_on_100_episodes.append(score)
#   epsilon=max(epsilon_ending_value,epsilon_decay_value*epsilon)
#   print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon : {:.2f}'.format(episode,np.mean(scores_on_100_episodes), epsilon),end="")
#   if episode%100==0:
#     print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon : {:.2f}'.format(episode,np.mean(scores_on_100_episodes), epsilon))
#   if np.mean(scores_on_100_episodes)>=200.0:
#     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tEpsilon : {:.2f}'.format(episode-100,np.mean(scores_on_100_episodes), epsilon))
#     torch.save(agent.local_qnetwork.state_dict(),'checkpoint.pth')
#     break