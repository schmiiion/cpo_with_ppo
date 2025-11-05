import gymnasium as gym
import safety_gymnasium as sg
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        render_mode = "rgb_array" if capture_video else 'human'
        env = sg.make(gym_id, render_mode=render_mode)
        env = CostIntoInfo(env)
        env = RecordEpisodeStatistics(env, deque_size=1000) # adds "r" and "l" entries to info dict
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"logs/videos/{run_name}",
                episode_trigger=lambda ep: ep % 50 == 0)

        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk


class CostIntoInfo(gym.Wrapper):
    """
    Initial reason for this wrapper:
        Adds the cost to the info, such that the RecordVideo Wrapper does not freak out
    Safety Gymansium steps: obs, reward, cost, terminated, truncated, info
    This wrapper hdes "cost" from the signature and always puts:
        info["cost"]        :scalar step cost
        info["cost_sum"]    :running episode cost sum
    Use this on single envs BEFORE vectorization
    """
    def __init__(self, env):
        super().__init__(env)
        self._cost_sum = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cost_sum = 0.0
        info = dict(info) if info is not None else {}
        #enusre that keys exist at time 0
        info["cost"] = np.float32(0.0)
        info["cost_sum"] = np.float32(self._cost_sum)
        return obs, info

    def step(self, action):
        obs, r, c, terminated, truncated, info = self.env.step(action)

        #accumulate cost
        self._cost_sum += float(c)

        info = dict(info) if info is not None else {}
        #preserve any existing keys
        info["cost"] = np.float32(c)
        info["cost_sum"] = np.float32(self._cost_sum)

        if terminated or truncated:
            info["final_cost_sum"] = np.float32(self._cost_sum)
            self._cost_sum = 0.0

        return obs, r, terminated, truncated, info