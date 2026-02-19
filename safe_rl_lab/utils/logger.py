import wandb
from omegaconf import OmegaConf, DictConfig
import time


class Logger:
    def __init__(self, cfg: DictConfig, is_debugging):
        self._is_active = not is_debugging
        self.cfg = cfg

        if not self.is_active:
            print("Logger initialized in SILENT mode. No data will be uploaded.")
            return

        #convert to standard dict
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        if cfg.algo.name == "PPG" or cfg.algo.name == "PPG_Lag": #they cant choose an architecture
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
        """
        Central logging function.

        Args:
            metrics: Dictionary of {metric_name: value}
            step: The global training step (optional but recommended)
            prefix: Optional string to prepend to metric names (e.g. "train/", "eval/")
        """
        if not self._is_active:
            return

        if prefix:
            # "loss" -> "train/loss"
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if self.cfg.algo.use_cost:
            cost_limit = self.cfg.algo.cost_limit * self.cfg.algo.cost_scaling
            assert cost_limit == 25
            metrics["rollout/cost_limit"] = cost_limit

        wandb.log(metrics, step=step)

    def close(self):
        if not self._is_active:
            return
        """Finish the run"""
        wandb.finish()