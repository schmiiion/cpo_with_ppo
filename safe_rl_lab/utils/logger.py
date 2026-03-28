import wandb
from omegaconf import OmegaConf, DictConfig
import time
import numpy as np
from collections import deque


class Logger:
    def __init__(self, cfg: DictConfig, is_debugging, window_size=150):
        self._is_active = not is_debugging
        self.cfg = cfg

        self._reward_window = deque(maxlen=window_size)
        self._cost_window = deque(maxlen=window_size)

        if not self._is_active:
            print("Logger initialized in SILENT mode. No data will be uploaded to WandB.")
            # We no longer 'return' here, because we still need to initialize
            # the rest of the class for Optuna's internal tracking!
            pass
        else:
            # convert to standard dict
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            if cfg.algo.name == "PPG" or cfg.algo.name == "PPG_Lag": # they cant choose an architecture
                self.run_name = f"{cfg.algo.name}-{cfg.env.gym_id}-{int(time.time())}"
            else:
                self.run_name = f"{cfg.algo.name}-{cfg.env.gym_id}-{cfg.algo.a2c_architecture}-{int(time.time())}"

            self.run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project_name,
                name=self.run_name,
                config=config_dict,
                monitor_gym=True, #TODO: True -> Videos logged
                save_code=True,
            )

    @property
    def is_active(self):
        return self._is_active

    def log(self, metrics: dict, step: int = None, prefix: str = ""):
        """Central logging function."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if "rollout/rew" in metrics:
            self._reward_window.append(metrics["rollout/rew"])

        if "rollout/raw_cost" in metrics:
            self._cost_window.append(metrics["rollout/raw_cost"])

        if not self._is_active:
            return

        if self.cfg.algo.use_cost:
            metrics["rollout/cost_limit"] = 25

        wandb.log(metrics, step=step)

    def close(self):
        if not self._is_active:
            return
        """Finish the run"""
        wandb.finish()

    def get_smoothed_reward(self):
        if len(self._reward_window) == 0:
            return None
        return np.mean(self._reward_window)

    def get_smoothed_cost(self):
        if len(self._cost_window) == 0:
            return None
        return np.mean(self._cost_window)