from codebase.AgentRainbowDQN.RainbowDQN import Agent
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Not enough argument : python3 -m codebase.train <model_name> <nb_episode> <seed>")
        exit(84)

    try:
        nb_episode = int(sys.argv[2])
        seed = int(sys.argv[3])
    except ValueError:
        print("Usage : python3 -m codebase.train <model_name> <nb_episode> <seed>")
        print("<nb_episode> and <seed> must be valid number")
        exit(84)


    model = Agent(
        hyperparameter_set=sys.argv[1],
        nb_episode=nb_episode if nb_episode > 0 else 20000,
        seed=seed if seed >= 0 else 42
    )

    model.run()
