import argparse
from distutils.util import strtobool
import time
import torch
import wandb
import gymnasium
import random
import numpy as np

# OWN
from safe_rl_lab.envs.wrappers import make_env
from safe_rl_lab.algo.ppo_lag import PPOLag




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_id', type=str, default='SafetyPointGoal2-v0')
    # parser.add_argument('--gym_id', type=str, default='SafetyPointGoal0-v0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if enabled, the experiment will be tracked to WandB')
    parser.add_argument('--wandb-project-name', type=str, default='ppo_in_sg')
    parser.add_argument('--wandb-entity', type=str, default="safe_rl_lab")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--store-model', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    model_arch = "shared"
    squash_actions = True
    run_name = f"PPO-{args.gym_id}-{model_arch}-{args.lr}-{int(time.time())}"
    if args.track:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=run_name,
            config={
                "algo": "PPO_Lag",
                "env_id": args.gym_id,
                "seed": args.seed,
            },
            monitor_gym=False, #TODO: True -> Videos logged
            save_code=False,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cpu')

    # envs = gymnasium.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, 0.99)
    #      for i in range(args.num_envs)]
    # )
    envs = gymnasium.vector.AsyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, 0.99)
         for i in range(args.num_envs)], shared_memory=False
    )
    ppo_lag = PPOLag(envs, run_name=run_name, model_arch=model_arch, store_model=args.store_model)
    assert isinstance(envs.single_action_space, gymnasium.spaces.Box), "only continuous action space is supported"

    ppo_lag.train()

    # print("envs.single_observation_space.shape:", envs.single_observation_space.shape)
    # print("envs.single_action_space.shape:", envs.single_action_space.shape)