from game_model import Point, BallState, TableState, GameState
from strategy.strategy import get_actions, visualize_strategy

if __name__ == '__main__':
    table_state = TableState({
        BallState(Point(1, 0.8), 'cue'),
        BallState(Point(1.685, 0.504), 'solid'),
        BallState(Point(0.1, 0.1), 'solid'),
        BallState(Point(1.5, 0.6), 'stripes'),
        BallState(Point(0.3, 0.3), 'stripes'),
        BallState(Point(1.374, 0.393), 'eight'),
    })

    game_state = GameState(table_state, 'solid')

    actions = get_actions(game_state)[:2]
    print(actions)
    print(table_state)
    visualize_strategy(game_state, actions)
