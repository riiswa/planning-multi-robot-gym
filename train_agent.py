import os 
import math
import argparse
import numpy as np
import stable_baselines3

import planning_multi_robot_gym

from stable_baselines3 import PPO
from planning_multi_robot_gym.planning_multi_robot_env import PlanningMultiRobotEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_robots", type=int, required=False, default=1)
    parser.add_argument("--n_barriers", type=int, required=False, default=50)
    parser.add_argument("--training_timesteps", type=int, required=False, default=100000)
    parser.add_argument("--seed", type=int, required=False, default=42)

    args = parser.parse_args()

    print(stable_baselines3.__version__)

    experiment_name = f"PPO_agent_{args.n_robots}_robots_{args.n_barriers}_barriers"
    agent_name = f"PPO_{int(args.training_timesteps/1000)}k_steps_seed_{args.seed}"
    models_dir = os.path.join("models", experiment_name)
    logs_dir = os.path.join("logs", experiment_name)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = PlanningMultiRobotEnv(n_robots=args.n_robots, n_barriers=args.n_barriers, dict_action_space=False) #render_mode="None",
    model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed, tensorboard_log=logs_dir)

    model.learn(total_timesteps=args.training_timesteps, tb_log_name=agent_name)
    model.save(os.path.join(models_dir, agent_name))

        