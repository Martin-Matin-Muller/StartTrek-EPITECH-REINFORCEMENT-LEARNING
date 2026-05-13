from codebase.AgentRainbowDQN.RainbowDQN import AgentRainbowDQN
from codebase.AgentDDQN_PER_Dueling_N_Step_Noisy.DDQN_PER_Dueling_N_Step_Noisy import AgentDDQN_PER_Dueling_N_Step_Noisy
from codebase.AgentDDQN_PER_Dueling_N_Step.DDQN_PER_Dueling_N_Step import AgentDDQN_PER_Dueling_N_Step
from codebase.AgentDDQN_PER_Dueling.DDQN_PER_Dueling import AgentDDQN_PER_Dueling
from codebase.AgentDDQN_PER.DDQN_PER import AgentDDQN_PER
from codebase.AgentDDQN.DDQN import AgentDDQN
from codebase.AgentDQN.DQN import AgentDQN
from codebase.AgentPPO.PPO import AgentPPO
from codebase.AgentHuman.Human import AgentHuman
from codebase.AgentHeuristic.Heuristic import AgentHeuristic
from codebase.AgentQLearning.QLearning import AgentQLearning
import sys

def choose_model():
    print("\n==============================")
    print("   RL Model Selection")
    print("==============================\n")
    
    print("1. DQN")
    print("2. DDQN")
    print("3. DDQN + PER")
    print("4. DDQN + PER + Dueling")
    print("5. DDQN + PER + Dueling + N-Step")
    print("6. DDQN + PER + Dueling + N-Step + Noisy Net")
    print("7. Rainbow DQN")
    print("8. PPO")
    print("9. Manual model")
    print("10. Heuristic")
    print("11. QLearning\n")

    while True:
        try:
            choice = int(input("-> Enter your choice (1-11): "))
            if 1 <= choice <= 11:
                break
            else:
                print("Invalid choice, please enter a number between 1 and 11.")
        except ValueError:
            print("Please enter a valid number.")

    models = {
        1: "DQN",
        2: "DDQN",
        3: "DDQN + PER",
        4: "DDQN + PER + Dueling",
        5: "DDQN + PER + Dueling + N-Step",
        6: "DDQN + PER + Dueling + N-Step + Noisy Net",
        7: "Rainbow DQN",
        8: "PPO",
        9: "Manual mode",
        10: "Heuristic",
        11: "QLearning"
    }

    print(f"\nYou selected: {models[choice]}\n")
    return choice


# shell or add an argmument
if __name__ == '__main__':
    models = {
        1: lambda nb_episode, seed: AgentDQN(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        2: lambda nb_episode, seed: AgentDDQN(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        3: lambda nb_episode, seed: AgentDDQN_PER(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        4: lambda nb_episode, seed: AgentDDQN_PER_Dueling(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        5: lambda nb_episode, seed: AgentDDQN_PER_Dueling_N_Step(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        6: lambda nb_episode, seed: AgentDDQN_PER_Dueling_N_Step_Noisy(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        7: lambda nb_episode, seed: AgentRainbowDQN(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        8: lambda nb_episode, seed: AgentPPO(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        9: lambda nb_episode, seed: AgentHuman(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        10: lambda nb_episode, seed: AgentHeuristic(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
        11: lambda nb_episode, seed: AgentQLearning(
                                        hyperparameter_set=sys.argv[1],
                                        nb_episode=nb_episode if nb_episode > 0 else 20000,
                                        seed=seed if seed >= 0 else 42),
                                    }

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

    choice = choose_model()
    model = models[choice](nb_episode, seed)

    # model.run(visualization=True)
    model.run()
