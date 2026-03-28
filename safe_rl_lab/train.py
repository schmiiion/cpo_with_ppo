import hydra
import gymnasium
from omegaconf import DictConfig
import sys
import optuna
import copy
import ast
import optunahub

from safe_rl_lab.envs.wrappers import make_env
from safe_rl_lab.factories.algo_factory import AlgoFactory
import torch

from safe_rl_lab.utils.logger import Logger
from safe_rl_lab.utils.other_stuff import seed_all


def run_single_experiment(cfg: DictConfig, trial: optuna.Trial = None):
    """Core logic for setting up and running a single experiment."""
    is_debugging = sys.gettrace() is not None
    seed_all(cfg.seed)

    logger = Logger(cfg, is_debugging=is_debugging)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if logger.is_active:
        run_name = logger.run_name
    else:
        run_name = "placeholder"

    env = gymnasium.vector.AsyncVectorEnv(
        [make_env(cfg.env.gym_id, cfg.seed + i, i, cfg.env.capture_video, run_name, 0.99)
         for i in range(cfg.env.num_envs)], shared_memory=True
    )

    algo = AlgoFactory.create(cfg, env, logger, device)

    #Run training
    algo.learn()

    final_reward = float(algo.logger.get_smoothed_reward())
    final_cost = float(algo.logger.get_smoothed_cost())
    final_violation = max(0.0, final_cost -cfg.algo.cost_limit)

    env.close()
    logger.close()

    return final_reward, final_violation



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    is_optimize = cfg.get("optimize", False)

    if not is_optimize:
        print("Run SINGLE experiment...")
        run_single_experiment(cfg)

    else:
        print("Run HYPERPARAM Sweep...")

        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)

            # --- PPG Epochs & Phases ---
            trial_cfg.algo.N_pi = trial.suggest_categorical("N_pi", [8, 16, 32])
            trial_cfg.algo.E_pi = trial.suggest_int("E_pi", 1, 5)
            trial_cfg.algo.E_v = trial.suggest_int("E_v", 1, 5)
            trial_cfg.algo.E_aux = trial.suggest_int("E_aux", 3, 9)
            trial_cfg.algo.beta_clone = trial.suggest_float("beta_clone", 0.5, 2.0)

            # --- Architecture ---
            arch_string = trial.suggest_categorical("hidden_dims", ["[256, 256]", "[512, 512]"])
            trial_cfg.algo.hidden_sizes = ast.literal_eval(arch_string)

            # --- PID & Cost Constraints ---
            # Scalar mit dem ich den Cost MSE im Aux. Loss balanciere
            trial_cfg.algo.cost_loss_coef = trial.suggest_float("cost_loss_coef", 0.1, 2.0, log=True)
            # PID related
            trial_cfg.algo.k_i = trial.suggest_float("k_i", 1e-4, 1e-1, log=True)
            trial_cfg.algo.k_p = trial.suggest_float("k_p", 1e-3, 1.0, log=True)
            trial_cfg.algo.k_d = trial.suggest_float("k_d", 0.0, 0.05)

            # --- Core PPO ---
            trial_cfg.algo.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            trial_cfg.algo.entropy_coef = trial.suggest_float("entropy_coef", 0.0001, 0.01, log=True)
            trial_cfg.algo.max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0, 5.0])

            return run_single_experiment(trial_cfg, trial)

        # Load the AutoSampler from the OptunaHub registry
        module = optunahub.load_module(package="samplers/auto_sampler")
        smart_sampler = module.AutoSampler()

        storage_name = "sqlite:///logs/cppgpid_study.db"

        study = optuna.create_study(
            study_name="cppgpid_cargoal1_search",
            storage=storage_name,
            directions=["maximize", "minimize"],
            sampler=smart_sampler,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=75)


if __name__ == "__main__":
    main()