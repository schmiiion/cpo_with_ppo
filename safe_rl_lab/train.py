import hydra
import gymnasium
from omegaconf import DictConfig

from safe_rl_lab.envs.wrappers import make_env
from safe_rl_lab.factories.algo_factory import AlgoFactory
import torch

from safe_rl_lab.utils.logger import Logger


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = Logger(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Setup Env
    env  = gymnasium.vector.AsyncVectorEnv(
        [make_env(cfg.env.gym_id, cfg.seed + i, i, cfg.env.capture_video, logger.run_name, 0.99)
         for i in range(cfg.env.num_envs)], shared_memory=False
    )

    # 2. Setup Algorithm (including Agent)
    algo = AlgoFactory.create(cfg, env, logger, device)

    # 3. Run
    # No "runner" object needed. The algo drives itself.
    algo.learn()




if __name__ == "__main__":
    main()