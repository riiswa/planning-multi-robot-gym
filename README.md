# Planning Multi-Robot Gym

A [Gymnasium](https://gymnasium.farama.org/) environment for simulating multi-robot planning. This environment is an implementation of the task to plan for multiple robots to reach a target location, avoiding obstacles in their path. The implementation of the environment logic and graphics was based on this [code](https://www.doc.ic.ac.uk/~ajd/Robotics/RoboticsResources/planningmultirobot.py) written by [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

This environment simulates the movements of multiple robots in an environment with barriers and a target. The robots are controlled by the actions given by an agent. The observations received by the agent includes information about the position, velocity and orientation of each robot, the future position of the target and the future position of the obstacles. The goal of the agent is to navigate the robots to the target while avoiding collisions with the obstacles.

![](screenshot.gif)

## Installation

This package is not yet on PyPi so a local installation is required:

```commandline
git clone https://github.com/riiswa/planning-multi-robot-gym
cd planning-multi-robot-gym
pip install -e .
```



## Usage

```python
import planning_multi_robot_gym

# There are many adjustable parameters for this environment, please refer to the brief documentation in the code.
env = planning_multi_robot_gym.make("PlanningMultiRobot-v0", render_mode="human", n_robots=5, n_barriers=80)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = ... # Your policy
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Help wanted
Here is a non-exhaustive list of tasks for people who would be interested in contributing to this repository:

- Write documentation;
- Add tests;
- Create new agents;
- Improve performance (e.g. by vectorizing some calculations).
