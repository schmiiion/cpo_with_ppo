import gymnasium as gym
import safety_gymnasium as sg

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        render_mode = "rgb_array" if capture_video else 'human'
        env = sg.make(gym_id, render_mode=render_mode)
        env = CostIntoInfo(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda ep: ep % 1000 == 0)

        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk


class CostIntoInfo(gym.Wrapper):
    """Adds the cost to the info, such that the RecordVideo Wrapper does not freak out1"""
    def step(self, action):
        obs, r, c, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["cost"] = c
        return obs, r, terminated, truncated, info