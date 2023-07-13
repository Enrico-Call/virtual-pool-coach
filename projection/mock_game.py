from game_model import Action, Point, Trajectory, TableState, BallState, GameState
from strategy import compute_next_move
from typing import List

TABLE_STATES = [TableState(ball_states={
                        BallState(coordinate=Point(x=1.374, y=0.393), type='eight'),
                        BallState(coordinate=Point(x=1.685, y=0.504), type='solid'),
                        BallState(coordinate=Point(x=0.213, y=0.595), type='cue')}),
                TableState(ball_states={
                        BallState(coordinate=Point(x=1.374, y=0.393), type='eight'),
                        BallState(coordinate=Point(x=1.685, y=0.504), type='solid'),
                        BallState(coordinate=Point(x=0.213, y=0.595), type='cue')}),
                TableState(ball_states={
                        BallState(coordinate=Point(x=1.374, y=0.393), type='eight'),
                        BallState(coordinate=Point(x=1.685, y=0.504), type='solid'),
                        BallState(coordinate=Point(x=0.213, y=0.595), type='cue')}),
                TableState(ball_states={
                        BallState(coordinate=Point(x=1.374, y=0.393), type='eight'),
                        BallState(coordinate=Point(x=1.685, y=0.504), type='solid'),
                        BallState(coordinate=Point(x=0.213, y=0.595), type='cue')}),
                TableState(ball_states={
                        BallState(coordinate=Point(x=1.374, y=0.393), type='eight'),
                        BallState(coordinate=Point(x=1.685, y=0.504), type='solid'),
                        BallState(coordinate=Point(x=0.213, y=0.595), type='cue')}),
            ]

def get_all_actions(table_state: TableState, turn="stripes") -> List:
    game_state = GameState(
        table_state=table_state,
        turn=turn
    )
    actions = compute_next_move(game_state)
    return actions
