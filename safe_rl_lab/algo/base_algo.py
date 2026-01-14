from abc import ABC, abstractmethod
import time
from collections import deque, defaultdict
import wandb
import numpy as np

class BaseAlgo(ABC):
    def __init__(self, runner, logger, cfg, device):
        self.runner = runner
        self.logger = logger
        self.cfg = cfg
        self.device = device

        self.ep_stats = defaultdict(lambda: deque(maxlen=100))


        self.last_log_time = 0
        self.sps_window = self.sps_window = deque(maxlen=5)

    @abstractmethod
    def learn(self):
        pass

    def _log_sps(self, global_step):
        if self.last_log_time == 0:
            self.last_log_time = time.time()
            return
        else:
            current_time = time.time()
            elapsed_time = current_time - self.last_log_time

            steps_diff =self.cfg.rollout_size
            self.sps_window.append(steps_diff / elapsed_time)
            avg_sps = np.mean(np.array(self.sps_window))

            if wandb.run:
                wandb.log({"charts/SPS_mean": avg_sps}, step=global_step)

            # Reset trackers
            self.last_log_time = current_time

    def _process_episodic_stats(self, ep_infos):
        """
        Ingests a list of info dicts and pushes them into the rolling buffer.
        Scalable: It automatically discovers new keys (like 'cost').
        """
        for info in ep_infos:
            # Standard Gym Keys
            if "episode" in info:
                self.ep_stats["return"].append(info["episode"]["r"])
                self.ep_stats["len"].append(info["episode"]["l"])

            # Safety Gym / Custom Keys (e.g. 'cost', 'hazard_hits')
            # If your env wrapper puts cost in info['cost'], we grab it here
            if "cost" in info:
                self.ep_stats["cost"].append(info["cost"])
            elif "episode_cost" in info:  # Different wrapper naming convention
                self.ep_stats["cost"].append(info["episode_cost"])

    def _get_mean_stats(self):
        """Returns a dict of averages for logging."""
        return {
            f"charts/ep_{k}_mean": np.mean(v)
            for k, v in self.ep_stats.items() if len(v) > 0
        }