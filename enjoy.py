import os
import math
import numpy as np
import argparse

import planning_multi_robot_gym

from stable_baselines3 import PPO
from example import planning_policy
from planning_multi_robot_gym.planning_multi_robot_env import PlanningMultiRobotEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_robots", type=int, required=False, default=5)
    parser.add_argument("--n_barriers", type=int, required=False, default=80)
    parser.add_argument("--rendering_timesteps", type=int, required=False, default=1000)
    parser.add_argument("--render_mode", type=str, required=False, default="human")
    # parser.add_argument("--algorithm", type=str, required=False, default="PPO")
    parser.add_argument("--training_timesteps", type=int, required=False, default=100000)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument('--rl_policy', action='store_true')
    parser.add_argument('--scripted-policy', dest='rl_policy', action='store_false')
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    

    if args.rl_policy:
        print("RL policy")
        env = PlanningMultiRobotEnv(render_mode=args.render_mode, n_robots=args.n_robots, n_barriers=args.n_barriers, dict_action_space=False)

        experiment_name = f"PPO_agent_{args.n_robots}_robots_{args.n_barriers}_barriers"
        agent_name = f"PPO_{int(args.training_timesteps/1000)}k_steps_seed_{args.seed}"
        models_dir = os.path.join("models", experiment_name)

        model = PPO.load(os.path.join(models_dir, agent_name), env)
    else:
        print("Sripted policy")
        env = planning_multi_robot_gym.make("PlanningMultiRobot-v0", render_mode=args.render_mode, n_robots=args.n_robots, n_barriers=args.n_barriers)
        

    observation, info = env.reset(seed=42)

    for t in range(args.rendering_timesteps):
        if args.rl_policy:
            action, _ = model.predict(observation)
        else:
            action = planning_policy(observation, info)
        
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
