from codebase.DDQN_PER_Dueling_agent.DDQN_PER_Dueling import DDQN_PER_Dueling
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        exit(84)
    dql = DDQN_PER_Dueling(
        hyperparameter_set=sys.argv[1],
        nb_episode=int(sys.argv[2]) or 20000,
        seed=int(sys.argv[3])
    )
    dql.run()