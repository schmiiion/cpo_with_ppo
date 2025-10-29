import argparse
from distutils.util import strtobool
import gymnasium.wrappers
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.optim as optim
import torch.nn as nn
import safety_gymnasium as sg
import tensorboard
from envs.wrappers import CostIntoInfo
from envs.wrappers import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_id', type=str, default='SafetyAntGoal0-v0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--timesteps', type=int, default=25000)
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if enabled, the experiment will be tracked to WandB')
    parser.add_argument('--wandb-project-name', type=str, default='cpo_with_ppg')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}-{args.timesteps}-{args.lr}-{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False, #TODO: True -> Videos logged
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    #SEEDING
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    device = torch.device('cpu')

    env_fun = make_env(args.gym_id, args.seed, 0 , args.capture_video, run_name)
    env = env_fun()

    ppo = PPO()
    ppo.train(env)









    #envs
    # envs = gymnasium.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
    #      for i in range(args.num_envs)]
    # )
    # print("envs.single_observation_space.shape:", envs.single_observation_space.shape)
    # print("envs.single_action_space.shape:", envs.single_action_space.shape)