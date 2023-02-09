import math
import numpy as np

import planning_multi_robot_gym


def planning_policy(observation, info, forward_weight=12, obstacle_weight=6666):
    def calculate_closest_obstacle_distance(robot_position, barriers_positions):
        return np.sum(np.sqrt((barriers_positions - robot_position) ** 2), axis=1).min() - \
               info["barrier_radius"] - info["robot_radius"]

    action = []
    barrier_positions = observation["future_obstacle_positions"]
    for r in range(info["n_robots"]):
        best_benefit = -np.inf

        vL = observation["vL"][r]
        vR = observation["vR"][r]
        x, y = observation["robot_positions"][r]
        theta = observation["theta"][r]

        vL_possible_array = (vL - info["max_acceleration"] * info["dt"], vL, vL + info["max_acceleration"] * info["dt"])
        vR_possible_array = (vR - info["max_acceleration"] * info["dt"], vR, vR + info["max_acceleration"] * info["dt"])

        vL_chosen = 0
        vR_chosen = 0
        x_chosen = 0
        y_chosen = 0

        for vL_possible in vL_possible_array:
            for vR_possible in vR_possible_array:
                if info["max_velocity"] >= vL_possible >= -info["max_velocity"] and \
                        info["max_velocity"] >= vR_possible >= -info["max_velocity"]:
                    (x_predict, y_predict, theta_predict, _) = \
                        info["predict_position"](vL_possible, vR_possible, x, y, theta, info["tau"])
                    distance_to_obstacle = calculate_closest_obstacle_distance(
                        np.array([x_predict, y_predict]),
                        barrier_positions,
                    )
                    previous_target_distance = math.sqrt(
                        (x - observation["future_target_position"][0]) ** 2 +
                        (y - observation["future_target_position"][1]) ** 2
                    )
                    new_target_distance = math.sqrt(
                        (x_predict - observation["future_target_position"][0]) ** 2 +
                        (y_predict - observation["future_target_position"][1]) ** 2
                    )
                    distance_forward = previous_target_distance - new_target_distance
                    distance_benefit = forward_weight * distance_forward
                    if distance_to_obstacle < info["robot_radius"]:
                        obstacle_cost = obstacle_weight * (info["robot_radius"] - distance_to_obstacle)
                    else:
                        obstacle_cost = 0.

                    benefit = distance_benefit - obstacle_cost
                    if benefit > best_benefit:
                        vL_chosen = vL_possible
                        vR_chosen = vR_possible
                        x_chosen = x_predict
                        y_chosen = y_predict

                        best_benefit = benefit
        action.append([vL_chosen, vR_chosen])
        barrier_positions = np.vstack((barrier_positions, np.array([x_chosen, y_chosen])))
    action = np.stack(action)

    return {"vL": action[:, 0], "vR": action[:, 1]}


if __name__ == "__main__":
    env = planning_multi_robot_gym.make("PlanningMultiRobot-v0", render_mode="human", n_robots=5, n_barriers=80)

    observation, info = env.reset(seed=42)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(planning_policy(observation, info))
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
