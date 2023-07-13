import plotly.graph_objects as go
from game_model import BALL_RADIUS, TABLE_DIMENSIONS, POCKET_COORDINATES


def draw_circle(fig, coordinates, color, r=BALL_RADIUS):
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=(coordinates[0] - r), y0=(coordinates[1] - r), x1=(coordinates[0] + r), y1=(coordinates[1] + r),
                  line_color=color, fillcolor=color,
                  )


def draw_board(fig, target_balls, opponent_balls, cue_ball, eight_ball, dpm=500):
    # Set axes properties
    fig.update_xaxes(range=[0, TABLE_DIMENSIONS[0]], zeroline=False, constrain='domain')
    fig.update_yaxes(range=[0, TABLE_DIMENSIONS[1]], constrain='domain')

    # Add circles
    for ball in target_balls:
        draw_circle(fig, ball, 'red')
    for ball in opponent_balls:
        draw_circle(fig, ball, 'blue')
    draw_circle(fig, cue_ball, 'white')
    draw_circle(fig, eight_ball, 'black')
    for pocket in POCKET_COORDINATES:
        draw_circle(fig, pocket, 'black', r=0.03)

    # Set figure size
    fig.update_layout(
        width=TABLE_DIMENSIONS[0] * dpm,
        height=TABLE_DIMENSIONS[1] * dpm,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )


def draw_trajectory(fig, v1, v2, color='black'):
    fig.add_shape(type="line",
                  x0=v1[0], y0=v1[1], x1=v2[0], y1=v2[1],
                  line=dict(
                      color=color,
                      width=1,
                      dash="dash",

                  ),
                  )
    fig.add_trace(go.Scatter(x=[v1[0], v2[0]], y=[v1[1], v2[1]], mode="markers"))
    """fig.add_shape(type="line",
                  x0=v2[0], y0=v2[1], x1=target[0], y1=target[1],
                  line=dict(
                      color=color,
                      width=1,
                      dash="dash",

                  ),
                  )
    fig.add_trace(go.Scatter(x=[v2[0], target[0]], y=[v2[1], target[1]], mode="markers"))"""


def draw_all_actions(fig, actions):
    for action in actions:
        for trajectory in action.predicted_trajectories:
            draw_trajectory(fig, trajectory.start_coordinate.to_array(), trajectory.end_coordinate.to_array())
