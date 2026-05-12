def fuel_reward(action, reward, fuel):
    if action == 1 or action == 3:
        fuel -= 10
        reward -= 0.01
    if action == 2:
        fuel -= 16
        reward -= 0.1
    return reward, fuel

def fuel_end_reward(fuel):
    return ((fuel * fuel) / 50000) / 10