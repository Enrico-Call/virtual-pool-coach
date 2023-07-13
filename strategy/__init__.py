from typing import Tuple, List
import numpy as np

from game_model import GameState, Trajectory, Point, Action
from .strategy import get_actions


def compute_next_move(game_state: GameState) -> List[Action]:
    return get_actions(game_state)
