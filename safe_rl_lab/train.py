import hydra
from safe_rl_lab.factories.algo_factory import AlgoFactory

@hydra.main(config_name="config")
def main(cfg):
    # 1. Setup hardware (Env + Agent)
    # You might create these here or let the Factory handle agent creation
    env = ...

    # 2. Factory creates the Algorithm
    # The factory injects the PIDLagrange object into PPOLag here
    algo = AlgoFactory.create(cfg, env)

    # 3. Run
    # No "runner" object needed. The algo drives itself.
    algo.learn()