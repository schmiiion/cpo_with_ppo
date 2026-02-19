import hydra
import gymnasium
from omegaconf import DictConfig
import sys

from safe_rl_lab.envs.wrappers import make_env
from safe_rl_lab.factories.algo_factory import AlgoFactory
import torch

from safe_rl_lab.utils.logger import Logger
from safe_rl_lab.utils.other_stuff import seed_all


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    is_debugging = sys.gettrace() is not None

    seed_all(cfg.seed)

    logger = Logger(cfg, is_debugging=is_debugging)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Setup Env
    if logger.is_active:
        run_name = logger.run_name
    else:
        run_name = "placeholder"
    env  = gymnasium.vector.AsyncVectorEnv(
        [make_env(cfg.env.gym_id, cfg.seed + i, i, cfg.env.capture_video, run_name, 0.99)
         for i in range(cfg.env.num_envs)], shared_memory=True
    )
    # env = gymnasium.vector.SyncVectorEnv(
    #     [make_env(cfg.env.gym_id, cfg.seed + i, i, cfg.env.capture_video, run_name, 0.99)
    #      for i in range(cfg.env.num_envs)]
    # )

    # 2. Setup Algorithm (including Agent)
    algo = AlgoFactory.create(cfg, env, logger, device)

    # 3. Run
    # No "runner" object needed. The algo drives itself.
    algo.learn()




if __name__ == "__main__":
    main()