from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Set, Union, Tuple
import numpy as np
import numpy.typing as npt

# Turn value None is for the beginning of the game, when the players haven't picked sides yet
Turn = Literal['solid', 'stripes', None]
BallStateType = Literal['solid', 'stripes', 'eight', 'cue']
ActionForceType = Literal['soft', 'medium', 'hard', 'big force']


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_array(self) -> npt.NDArray:
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, array: npt.NDArray) -> Point:
        return cls(x=float(array[0]), y=float(array[1]))


@dataclass(frozen=True)
class BallState:
    """
    A ball (solid, stripes, the eight-ball and the white ball (cue ball)) consists of a point of coordinates.
    This class also contains the type of ball
    """
    coordinate: Point
    type: BallStateType


@dataclass(frozen=True)
class TableState:
    """
    The table is defined by a set of ball positions.
    """
    ball_states: Set[BallState]

    def is_game_over(self) -> bool:
        return all(ball.type != 'eight' for ball in self.ball_states)

    def has_ball_been_potted(self) -> bool:
        return len(self.ball_states) < 16

    def get_typed_ball_states(self, ball_state_type: BallStateType) -> List[BallState]:
        return [ball_state for ball_state in self.ball_states if ball_state.type == ball_state_type]

    def get_cue_ball_state(self) -> Union[BallState, None]:
        ball_state = self.get_typed_ball_states('cue')
        return ball_state[0] if ball_state else None

    def get_eight_ball_state(self) -> Union[BallState, None]:
        ball_state = self.get_typed_ball_states('eight')
        return ball_state[0] if ball_state else None


@dataclass(frozen=True)
class GameState:
    table_state: TableState
    turn: Turn

    @property
    def inverse_turn(self):
        if self.turn == 'solid':
            return 'stripes'
        elif self.turn == 'stripes':
            return 'solid'
        return None


@dataclass(frozen=True)
class Trajectory:
    """
    This class will store information about the (first segments of) the trajectory of a ball
    after it is hit by the cue stick or another ball. Which ball in the TableState corresponds
    to it should be identifiable through the start coordinate.
    Score determines the arbitrary difficulty of the selected shot (lower == easier)
    """
    start_coordinate: Point
    end_coordinate: Point

    def to_line_array(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.start_coordinate.to_array(), self.end_coordinate.to_array()


@dataclass(frozen=True)
class Action:
    """
    The action determines how to hit (the angle) the cue ball with the cue stick
    and the predicted ball trajectories. The first trajectory in the list is
    required to be the cue ball's trajectory.
    """
    force: ActionForceType
    predicted_trajectories: List[Trajectory]


BALL_RADIUS = 0.057/2
TABLE_DIMENSIONS = np.array([2.24, 1.12])
# Right hand rule DO NOT CHANGE!
POCKET_COORDINATES = [
    (0, 0),
    (TABLE_DIMENSIONS[0] / 2, 0),
    (TABLE_DIMENSIONS[0], 0),
    (0, TABLE_DIMENSIONS[1]),
    (TABLE_DIMENSIONS[0] / 2, TABLE_DIMENSIONS[1]),
    TABLE_DIMENSIONS
]
POCKET_RADIUS = 0.07


WIDTH = 500
HEIGHT = 700

LEFT_TOP = (226, 124)
LEFT_BOTTOM = (218, 545)
RIGHT_TOP = (1074, 135)
RIGHT_BOTTOM = (1066, 560)
