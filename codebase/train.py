import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human") # discrete by default
obs, info = env.reset(seed = 42)

while True :
    action = policy(obs) # random or learned
    obs, r, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()