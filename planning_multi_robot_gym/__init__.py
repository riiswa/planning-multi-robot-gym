from planning_multi_robot_gym.planning_multi_robot_env import PlanningMultiRobotEnv
from gymnasium import make
from gymnasium.envs.registration import register

register(
    id="PlanningMultiRobot-v0",
    entry_point="planning_multi_robot_gym:PlanningMultiRobotEnv"
)

__all__ = [
    make.__name__,
    PlanningMultiRobotEnv.__name__
]

