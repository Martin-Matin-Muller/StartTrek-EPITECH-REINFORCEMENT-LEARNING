from codebase.basic_agent.agent import Agent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument(
        '--train',
        nargs='?',
        const=1,
        type=int,
        metavar='N',
        help='Training mode (optionnel : nombre de runs)'
    )

    args = parser.parse_args()

    dql = Agent(
        hyperparameter_set=args.hyperparameters,
        nb_episode=args.train or 1
    )

    if args.train is not None:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)