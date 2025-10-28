import argparse
from torch.utils.tensorboard import SummaryWriter
import time
import safety_gymnasium
import tensorboard
import wandb
from algo import ppo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SafetyAntGoal0-v0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--timesteps', type=int, default=25000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run_name = f"{args.env}-{args.timesteps}-{args.lr}-{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)

    writer.flush()
    writer.close()

# env = safety_gymnasium.make('SafetyAntGoal0-v0', render_mode='human')
#
# obs, info = env.reset()
#
# ppo_alg = ppo.PPO()
# ppo_alg.train(env)

