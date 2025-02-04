import argparse
from train import train_dyna_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='dyna_ppo')
    args = parser.parse_args()

    train_dyna_ppo(
        args.env,
        exp_name=args.exp_name,
        seed=args.seed,
        epochs=args.epochs
    )
