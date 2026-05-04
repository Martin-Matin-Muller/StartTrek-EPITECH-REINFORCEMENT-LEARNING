from codebase.AgentRainbowDQN.RainbowDQN import Agent
import sys


import gymnasium as gym
import os
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# def dqn(hyperparameter_set, nb_episode=2000, seed=42, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
#     env = gym.make('LunarLander-v3', render_mode="rgb_array")
#     agent = Agent(hyperparameter_set=hyperparameter_set)
#     env.reset(seed=seed)
#     scores = []                        # list containing scores from each episode
#     scores_window = deque(maxlen=100)  # last 100 scores
#     eps = eps_start                    # initialize epsilon
#     for i_episode in range(1, nb_episode+1):
#         state,info = env.reset()
#         score = 0
#         for t in range(max_t):
#             action = agent.act(state, eps)
#             next_state, reward, done, info,_ = env.step(action)
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             score += reward
#             if done:
#                 break 
#         scores_window.append(score)       # save most recent score
#         scores.append(score)              # save most recent score
#         eps = max(eps_end, eps_decay*eps) # decrease epsilon
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
#         if i_episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#         if np.mean(scores_window)>=200.0:
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
#             torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_lunar.pth')
#             break
#     return scores


# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

def plot_scores(scores, save_path="scores.png"):

    if len(scores) == 0:
        print("Aucun score à afficher")
        return

    # Crée le dossier si nécessaire
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("Training Scores")

    # Sauvegarde
    plt.savefig(save_path)
    print(f"Graph sauvegardé ici : {save_path}")

    plt.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Not enough argument : python3 -m <model_name> <nb_episode> <seed>")
        exit(84)

    try:
        nb_episode = int(sys.argv[2])
        seed = int(sys.argv[3])
    except ValueError:
        print("Usage : python3 -m <model_name> <nb_episode> <seed>")
        print("<nb_episode> and <seed> must be valid number")
        exit(84)


    model = Agent(
        hyperparameter_set=sys.argv[1],
        nb_episode=nb_episode if nb_episode > 0 else 20000,
        seed=seed if seed >= 0 else 42
    )

    scores = model.run()
    plot_scores(scores=scores, save_path="runs/" + sys.argv[1] + "_detail_RainbowDQN.png")

    # dql = DDQN_PER_Dueling(
    #     hyperparameter_set=sys.argv[1],
    #     nb_episode=nb_episode if nb_episode > 0 else 20000,
    #     seed=seed if seed > 0 else 42
    # )
    # dql.run()

