from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from game_model import (GameState,
                        Trajectory,
                        Point,
                        Action,
                        ActionForceType,
                        BALL_RADIUS,
                        POCKET_COORDINATES,
                        TABLE_DIMENSIONS,
                        POCKET_RADIUS)
from strategy.draw import draw_board, draw_all_actions

TABLE_CORNERS: Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] = (  # right hand rule
    np.array([0, 0]),  # bottom left
    np.array([TABLE_DIMENSIONS[0], 0]),  # bottom right
    np.array([TABLE_DIMENSIONS[0], TABLE_DIMENSIONS[1]]),  # top right
    np.array([0, TABLE_DIMENSIONS[1]]),  # top left
)

CUSHION_LINES: List[Tuple[npt.NDArray, npt.NDArray]] = [
    (TABLE_CORNERS[0], TABLE_CORNERS[1]),
    (TABLE_CORNERS[1], TABLE_CORNERS[2]),
    (TABLE_CORNERS[2], TABLE_CORNERS[3]),
    (TABLE_CORNERS[3], TABLE_CORNERS[0])
]

ERROR_BOUND_POCKET_RADIUS = 0.01


def _add_value_to_dim(point: npt.NDArray, dim: int, val: float, weight=1):
    addition = np.zeros(len(point))
    addition[dim] = val
    return point + weight * addition


def _add_pocket_radius(point: npt.NDArray, dim: int, weight=1):
    return _add_value_to_dim(point, dim, POCKET_RADIUS+ERROR_BOUND_POCKET_RADIUS, weight)


def is_horizontal(line: Tuple[npt.NDArray, npt.NDArray]):
    return line[0][1] == line[1][1]


HALF_CUSHION_LINES: List[Tuple[npt.NDArray, npt.NDArray]] = [
    (_add_pocket_radius(TABLE_CORNERS[0], 0), _add_pocket_radius(POCKET_COORDINATES[1], 0, -1)),
    (_add_pocket_radius(POCKET_COORDINATES[1], 0), _add_pocket_radius(TABLE_CORNERS[1], 0, -1)),
    (_add_pocket_radius(TABLE_CORNERS[1], 1), _add_pocket_radius(TABLE_CORNERS[2], 1, -1)),
    (_add_pocket_radius(TABLE_CORNERS[2], 0, -1), _add_pocket_radius(POCKET_COORDINATES[4], 0)),
    (_add_pocket_radius(POCKET_COORDINATES[4], 0, -1), _add_pocket_radius(TABLE_CORNERS[2], 0)),
    (_add_pocket_radius(TABLE_CORNERS[3], 1, -1), _add_pocket_radius(TABLE_CORNERS[0], 1))
]


def euclidean_distance(v1: npt.NDArray, v2: npt.NDArray) -> np.ScalarType:
    return np.linalg.norm(v1 - v2)


def get_intersection(line1: Tuple[npt.NDArray, npt.NDArray],
                     line2: Tuple[npt.NDArray, npt.NDArray]) -> Union[npt.NDArray, None]:
    def line(p1, p2):
        return (p1[1] - p2[1]), (p2[0] - p1[0]), -(p1[0] * p2[1] - p2[0] * p1[1])

    line1_coef = line(line1[0], line1[1])
    line2_coef = line(line2[0], line2[1])

    d = line1_coef[0] * line2_coef[1] - line1_coef[1] * line2_coef[0]
    dx = line1_coef[2] * line2_coef[1] - line1_coef[1] * line2_coef[2]
    dy = line1_coef[0] * line2_coef[2] - line1_coef[2] * line2_coef[0]
    if d != 0:
        x = dx / d
        y = dy / d
        # if min(line1[0][0], line1[1][0])-0.00001 <= x <= max(line1[0][0], line1[1][0])+0.00001 and \
        #    min(line1[0][1], line1[1][1])-0.00001 <= y <= max(line1[0][1], line1[1][1])+0.00001 and \
        #    min(line2[0][0], line2[1][0])-0.00001 <= x <= max(line2[0][0], line2[1][0])+0.00001 and \
        #    min(line2[0][1], line2[1][1])-0.00001 <= y <= max(line2[0][1], line2[1][1])+0.00001:
        return np.array([x, y])

    return None


@dataclass()
class Shot:
    trajectories: List[Trajectory]
    theta: float

    def get_distances(self) -> List[float]:
        return [euclidean_distance(trajectory.start_coordinate.to_array(), trajectory.end_coordinate.to_array())
                for trajectory in self.trajectories]

    def get_total_distance(self) -> float:
        return sum(self.get_distances())

    def get_shot_difficulty(self) -> float:
        arbitrary_theta_weight = 2.0
        bounce_trajectories = self.get_bounce_trajectories()
        bounced_theta_sum = sum([np.abs(get_theta_trajectories(*trajectories)-np.pi) for trajectories in bounce_trajectories])
        return self.get_total_distance() + arbitrary_theta_weight * np.exp(np.abs(self.theta - np.pi)) + bounced_theta_sum

    def get_bounce_trajectories(self) -> List[Tuple[Trajectory, Trajectory]]:
        results: List[Tuple[Trajectory, Trajectory]] = []
        for i, trajectory in enumerate(self.trajectories[:-1]):
            if all(trajectory.end_coordinate.to_array() == self.trajectories[i+1].start_coordinate.to_array()):
                results.append((trajectory, self.trajectories[i+1]))
        return results

    def get_force(self) -> ActionForceType:
        if len(self.trajectories) == 1:
            # No pocket-able ball --> big force!
            return 'big force'
        distance = self.get_total_distance()
        if np.abs(self.theta - np.pi) > np.pi / 3 and distance > 0.5:
            # larger than 60-degree angle + large distance >0.5m
            return 'big force'
        elif np.abs(self.theta - np.pi) < np.pi / 4 and distance < 0.5:
            # smaller than 45-degree angle + small distance <0.5m
            return 'soft'
        return 'medium'

    def get_cue_trajectory(self) -> Trajectory:
        if not self.trajectories:
            raise ValueError("No trajectories set, at least one trajectory needed to compute cue trajectory.")
        # recenter trajectory to origin of cue ball
        trajectory_vector = (self.trajectories[0].end_coordinate.to_array() -
                             self.trajectories[0].start_coordinate.to_array())
        trajectory_vector = rotate_vector(trajectory_vector, np.pi)  # rotate 180 degrees
        # Move back to table coordinate system
        trajectory_vector += self.trajectories[0].start_coordinate.to_array()
        return Trajectory(start_coordinate=Point.from_array(trajectory_vector),
                          end_coordinate=self.trajectories[0].start_coordinate)

    def to_action(self) -> Action:
        return Action(force=self.get_force(), predicted_trajectories=self.trajectories)


def get_theta(v1: npt.NDArray, v2: npt.NDArray, center: npt.NDArray) -> np.ScalarType:
    return np.arccos(np.dot(v1 - center, v2 - center) / (np.linalg.norm(v1 - center) * np.linalg.norm(v2 - center)))


def get_theta_trajectories(trajectory1: Trajectory, trajectory2: Trajectory) -> np.ScalarType:
    return get_theta(
        v1=trajectory1.start_coordinate.to_array(),
        v2=trajectory2.end_coordinate.to_array(),
        center=trajectory1.end_coordinate.to_array()
    )


def calculate_distance_line2point(line: Tuple[npt.NDArray, npt.NDArray], p: npt.NDArray) -> np.ScalarType:
    return np.abs(np.cross(line[1] - line[0], line[0] - p)) / np.linalg.norm(line[1] - line[0])


def check_collision(trajectory: Trajectory, p: npt.NDArray) -> bool:
    arbitrary_error_bound = 0.005
    p1 = trajectory.start_coordinate.to_array()
    p2 = trajectory.end_coordinate.to_array()
    in_between_x = min(p1[0], p2[0]) < p[0] < max(p1[0], p2[0])
    in_between_y = min(p1[1], p2[1]) < p[1] < max(p1[1], p2[1])
    if not in_between_x and not in_between_y:
        return False
    dist = calculate_distance_line2point(trajectory.to_line_array(), p)
    return dist <= (2 * BALL_RADIUS + arbitrary_error_bound)


def check_collision_with_walls(trajectory: Trajectory):
    for cushion_line in HALF_CUSHION_LINES:
        if get_intersection(trajectory.to_line_array(), cushion_line) is not None:
            print(f"Intersection at {get_intersection(trajectory.to_line_array(), cushion_line)} for {trajectory}")
            return True
        if (
                calculate_distance_line2point(cushion_line, trajectory.start_coordinate.to_array()) < BALL_RADIUS or
                calculate_distance_line2point(cushion_line, trajectory.end_coordinate.to_array()) < BALL_RADIUS or
                calculate_distance_line2point(trajectory.to_line_array(), cushion_line[0]) < BALL_RADIUS or
                calculate_distance_line2point(trajectory.to_line_array(), cushion_line[1]) < BALL_RADIUS
        ):
            return True
    return False


def check_any_collision(trajectory: Trajectory, collision_balls: List[npt.NDArray]) -> bool:
    for collision_ball in collision_balls:
        if (trajectory.start_coordinate.to_array() == collision_ball).all():
            continue  # skip the current ball
        if check_collision(trajectory, collision_ball):
            return True
    # return check_collision_with_walls(trajectory)
    return False


def reflection_of_point(p_0, q_i, q_j):
    """Calculates reflection of a point across an edge

    Args:
        p_0 (ndarray): Inner point, (2,)
        q_i (ndarray): First vertex of the edge, (2,)
        q_j (ndarray): Second vertex of the edge, (2,)

    Returns:
        ndarray: Reflected point, (2,)
    Stolen from: https://stackoverflow.com/questions/6949722/reflection-of-a-point-over-a-line
    """

    a = q_i[1] - q_j[1]
    b = q_j[0] - q_i[0]
    c = - (a * q_i[0] + b * q_i[1])

    p_k = (np.array([[b ** 2 - a ** 2, -2 * a * b],
                     [-2 * a * b, a ** 2 - b ** 2]]) @ p_0 - 2 * c * np.array([a, b])) / (a ** 2 + b ** 2)

    return p_k


def get_cushion_mirrored_coordinates(coordinate: npt.NDArray) -> List[Tuple[npt.NDArray,
                                                                            Tuple[npt.NDArray, npt.NDArray]]]:
    """
        returns: [(mirrored_coordinate, mirror_line), ...]
    """
    results = []
    for cushion_line in CUSHION_LINES:
        if is_horizontal(cushion_line):
            dim = 1
            if cushion_line[0][0] == 0:
                weight = 1
            else:
                weight = -1

        else:
            dim = 0
            if cushion_line[0][1] == 0:
                weight = 1
            else:
                weight = -1

        mirror_line = (_add_value_to_dim(cushion_line[0], dim, BALL_RADIUS, weight),
                       _add_value_to_dim(cushion_line[1], dim, BALL_RADIUS, weight))
        results.append((reflection_of_point(coordinate, mirror_line[0], mirror_line[1]), mirror_line))
    return results


def get_bounced_trajectories(start_point: npt.NDArray, end_point: npt.NDArray) -> List[List[Trajectory]]:
    mirrored_coordinates = get_cushion_mirrored_coordinates(end_point)
    intersection_points = [get_intersection((start_point, mirrored_end_point), mirror_line)
                           for mirrored_end_point, mirror_line in mirrored_coordinates]

    result: List[List[Trajectory]] = []
    for mirrored_coordinate, intersection_point in zip(mirrored_coordinates, intersection_points):
        if intersection_point is None:
            continue

        # Check if bounce is in a pocket
        if any(euclidean_distance(intersection_point, pocket) <= POCKET_RADIUS for pocket in POCKET_COORDINATES):
            continue
        result.append([
            Trajectory(
                start_coordinate=Point.from_array(start_point),
                end_coordinate=Point.from_array(intersection_point)
            ),
            Trajectory(
                start_coordinate=Point.from_array(intersection_point),
                end_coordinate=Point.from_array(end_point)
            )])
    return result


def get_shots(cue_ball: npt.NDArray,
              target_balls: List[npt.NDArray],
              opponent_balls: List[npt.NDArray]) -> List[Shot]:
    results = []
    for i, ball in enumerate(target_balls):
        for pocket in POCKET_COORDINATES:

            ball_to_pocket_trajectories = [[Trajectory(
                start_coordinate=Point.from_array(ball),
                end_coordinate=Point.from_array(pocket),
            )]] + get_bounced_trajectories(ball, pocket)

            for ball_to_pocket_trajectory in ball_to_pocket_trajectories:
                trajectory_end_point = get_trajectory_end_point(ball,
                                                                ball_to_pocket_trajectory[0].end_coordinate.to_array())
                direct_trajectory = Trajectory(
                    start_coordinate=Point.from_array(cue_ball),
                    end_coordinate=Point.from_array(trajectory_end_point)
                )
                bounced_trajectories = get_bounced_trajectories(cue_ball, trajectory_end_point)

                for trajectories in bounced_trajectories + [[direct_trajectory]]:
                    theta = get_theta_trajectories(trajectories[-1], ball_to_pocket_trajectory[0])
                    if np.abs(theta - np.pi) > np.pi / 2:
                        continue

                    collision_balls = target_balls + opponent_balls
                    if any(check_any_collision(trajectory, collision_balls)
                           for trajectory in trajectories + ball_to_pocket_trajectory):
                        continue
                    shot = Shot(
                        trajectories=trajectories + ball_to_pocket_trajectory,
                        theta=theta
                    )
                    # shot.trajectories = [shot.get_cue_trajectory()] + shot.trajectories
                    results.append(shot)

    if not results:
        # No pocket-able shots, hit the closest target ball with the largest force
        for ball in sorted(target_balls, key=lambda b: euclidean_distance(cue_ball, b)):
            trajectory = Trajectory(
                start_coordinate=Point.from_array(cue_ball),
                end_coordinate=Point.from_array(ball)
            )
            collision_balls = (target_balls + opponent_balls)
            if check_any_collision(trajectory, collision_balls):
                continue
            return [Shot(
                trajectories=[trajectory],
                theta=0
            )]
    return sorted(results, key=lambda s: s.get_shot_difficulty())


def rotate_vector(v: npt.NDArray, theta: float) -> npt.NDArray:
    c = np.cos(theta)
    s = np.sin(theta)
    rotation_matrix = np.matrix([[c, -s], [s, c]])
    return np.array(np.dot(rotation_matrix, v)).flatten()


def get_trajectory_end_point(ball2: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
    target_c = target - ball2
    theta_to_target = np.arctan(target_c[1] / target_c[0]) if target_c[0] != 0 else 0
    if target_c[0] < 0:
        theta_to_target = theta_to_target - np.pi
    trajectory_end_point = np.array([2 * BALL_RADIUS, 0])
    trajectory_end_point = rotate_vector(trajectory_end_point, theta_to_target + np.pi)
    trajectory_end_point += ball2
    return trajectory_end_point


def get_actions(game_state: GameState) -> List[Action]:
    if game_state.table_state.is_game_over():
        return []
    cue_ball = game_state.table_state.get_cue_ball_state().coordinate.to_array()
    eight_ball = game_state.table_state.get_eight_ball_state().coordinate.to_array()
    if game_state.turn is None:
        # No ball has been pocketed yet
        target_balls = (game_state.table_state.get_typed_ball_states('solid') +
                        game_state.table_state.get_typed_ball_states('stripes'))
        opponent_balls = []
    else:
        target_balls = game_state.table_state.get_typed_ball_states(game_state.turn)
        opponent_balls = game_state.table_state.get_typed_ball_states(game_state.inverse_turn)

    target_balls = [target_ball.coordinate.to_array() for target_ball in target_balls]
    opponent_balls = [opponent_ball.coordinate.to_array() for opponent_ball in opponent_balls]
    if not target_balls:
        target_balls += [eight_ball]
    else:
        opponent_balls += [eight_ball]
    shots = get_shots(cue_ball, target_balls, opponent_balls)
    return [shot.to_action() for shot in shots]


def visualize_strategy(game_state, actions):
    fig = go.Figure()
    target_balls = game_state.table_state.get_typed_ball_states(game_state.turn)
    opponent_balls = game_state.table_state.get_typed_ball_states(game_state.inverse_turn)

    target_balls = [target_ball.coordinate.to_array() for target_ball in target_balls]
    opponent_balls = [opponent_ball.coordinate.to_array() for opponent_ball in opponent_balls]
    cue_ball = game_state.table_state.get_cue_ball_state().coordinate.to_array()
    eight_ball = game_state.table_state.get_eight_ball_state().coordinate.to_array()

    draw_board(fig, target_balls, opponent_balls, cue_ball, eight_ball)
    draw_all_actions(fig, actions)

    fig.show()
