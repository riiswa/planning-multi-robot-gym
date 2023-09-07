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
    parser.add_argument("--episode_length", type=int, required=False, default=None)

    args = parser.parse_args()

    # Create files to log the training results and the final model
    experiment_name = f"PPO_agent_{args.n_robots}_robots_{args.n_barriers}_barriers"
    agent_name = f"PPO_{int(args.training_timesteps/1000)}k_steps_seed_{args.seed}"
    models_dir = os.path.join("models", experiment_name)
    logs_dir = os.path.join("logs", experiment_name)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Configure the agent and the environment with the desired parameters
    env = PlanningMultiRobotEnv(n_robots=args.n_robots, n_barriers=args.n_barriers, episode_length=args.episode_length, dict_action_space=False)
    model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed, tensorboard_log=logs_dir)

    # Train and save the agent 
    model.learn(total_timesteps=args.training_timesteps, tb_log_name=agent_name)
    model.save(os.path.join(models_dir, agent_name))

        