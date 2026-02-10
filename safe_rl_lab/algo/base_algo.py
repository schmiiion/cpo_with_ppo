from abc import ABC, abstractmethod
import time
from collections import deque, defaultdict
import numpy as np
from safe_rl_lab.utils.logger import Logger

class BaseAlgo(ABC):
    def __init__(self, runner, logger, cfg, device):
        self.runner = runner
        self.logger: Logger = logger
        self.cfg = cfg
        self.device = device
        self.global_step = 0

        self.rollout_rolling_stats = defaultdict(lambda: deque(maxlen=100))

        self.last_log_time = 0
        self.sps_window = deque(maxlen=5)

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

            steps_diff =self.cfg.algo.rollout_size
            self.sps_window.append(steps_diff / elapsed_time)
            avg_sps = np.mean(np.array(self.sps_window))

            if self.logger.is_active:
                self.logger.log(metrics={"charts/SPS_mean": avg_sps}, step=global_step)

            # Reset trackers
            self.last_log_time = current_time

    def _process_episodic_stats(self, rollout_stats):
        """
        Ingests a list of info dicts and pushes them into the rolling buffer.
        Scalable: It automatically discovers new keys (like 'cost').
        """
        #skip if no trajectories finished during the rollout
        if not rollout_stats:
            return

        #push new data in rolling buffer
        for k, v in rollout_stats.items():
            self.rollout_rolling_stats[k].append(v)

        #create return dict
        rolling_metrics = {}
        for k, deque_values in self.rollout_rolling_stats.items():
            if len(deque_values) > 0:
                rolling_metrics[f"{k}"] = np.mean(np.array(deque_values))

        #Log the smoothed values
        self.logger.log(rolling_metrics, step=self.global_step, prefix="rollout")

    def _get_mean_stats(self):
        """Returns a dict of averages for logging."""
        return {
            f"charts/ep_{k}_mean": np.mean(v)
            for k, v in self.rollout_rolling_stats.items() if len(v) > 0
        }

    def _safe_if_best(self, global_step):
        pass