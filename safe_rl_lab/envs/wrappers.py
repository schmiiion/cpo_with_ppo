import gymnasium as gym
import safety_gymnasium as sg
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


def make_env(gym_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        render_mode = "rgb_array" if capture_video else 'human'
        env = sg.make(gym_id, render_mode=render_mode)
        env = CostIntoInfo(env)
        env = gym.wrappers.FlattenObservation(env)
        env = RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0: #TODO hardcoded video omitting
            env = gym.wrappers.RecordVideo(
                env,
                f"logs/videos/{run_name}",
                episode_trigger=lambda ep: ep % 30 == 0)
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
        # print("#################################")
        # print("wrapper created")
        # print("#################################")

    def reset(self, **kwargs):
        # print("-"*20)
        # print("wrapper RESET")
        # print("-"*20)
        obs, info = self.env.reset(**kwargs)
        self._cost_sum = 0.0
        info["cost"] = np.float32(0.0)
        return obs, info

    def step(self, action):
        obs, r, c, terminated, truncated, info = self.env.step(action)

        #accumulate cost
        self._cost_sum += float(c)

        assert c == info["cost_sum"]
        info["cost"] = np.float32(c)

        if terminated or truncated:
            # print('o'*30)
            info["acc_cost"] = np.float32(self._cost_sum)
            self._cost_sum = 0.0

        return obs, r, terminated, truncated, info