import re
import torch

# OWN
from safe_rl_lab.envs.wrappers import make_env
from safe_rl_lab.models.agent import Agent
from safe_rl_lab.models.sharedBackboneAgent import SharedBackboneAgent



if __name__ == '__main__':
    run_name = f"checkpoints/SafetyRacecarGoal2-v0-shared-0.0003-1763299753_best.pt"
    env_id = re.search(r'(?<=/)([^-]+-[^-]+)', run_name).group(1)
    print(f"env_id: {env_id}")

    capture_video = False
    env = make_env(env_id, 1, 1,capture_video, run_name, 0.99)()

    hidden_dim = 64
    if "shared" in run_name:
        agent = SharedBackboneAgent(env, hidden_dim)
    elif "separate" in run_name:
        pass
    else:
        pass

    # Load check<point
    checkpoint = torch.load(run_name, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()

    num_episodes = 10  # or float("inf") to run forever
    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        done = False
        ep_reward, ep_cost, steps = 0.0, 0.0, 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            a = action.squeeze(0).cpu().numpy()
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            ep_reward += r
            if "cost" in info:
                ep_cost += info["cost"]
            steps += 1

            if not capture_video:
                env.render()

            obs = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, cost={ep_cost:.2f}, steps={steps}")

    env.close()


