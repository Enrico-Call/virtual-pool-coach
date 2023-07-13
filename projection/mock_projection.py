from tkinter import Tk, Canvas, Frame, mainloop, BOTH, NONE, Toplevel
from game_model import Point, BallState, Trajectory, TableState, Action
class Projection(Frame):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.canvas = Canvas(self.parent)
        self.draw_grid(10)

    def draw_lines(self, start_coords, end_coords, color="red"):
        for start_coord, end_coord in zip(start_coords, end_coords):
            self.canvas.create_line(start_coord, end_coord, fill=color)
        self.canvas.pack(fill="both", expand=True)

    def draw_circles(self, coords, radiuses):
        for radius, center_coord in zip(radiuses, coords):
            coord1, coord2 =  self.get_circle_coords(center_coord, radius)
            self.canvas.create_oval(coord1, coord2)
        self.canvas.pack(fill="both", expand=True)

    def get_circle_coords(self, center_coord, radius):
        x0, y0 = [coord - radius for coord in center_coord]
        x1, y1 = [coord + radius for coord in center_coord]
        return [x0, y0], [x1, y1]

    def draw_grid(self, line_distance):
        self.parent.update()
        width = self.parent.winfo_width()
        height = self.parent.winfo_height()
        horizontal_start_coords = [[0, i] for i in range(0, height, line_distance)]
        horizontal_end_coords = [[width, i] for i in range(0, height, line_distance)]
        self.draw_lines(horizontal_start_coords, horizontal_end_coords, "black")
        vertical_start_coords = [[i, 0] for i in range(0, width, line_distance)]
        vertical_end_coords = [[i, height] for i in range(0, width, line_distance)]
        self.draw_lines(vertical_start_coords, vertical_end_coords, "black")

def main():
    # specify resolutions of both windows
    w0, h0 = 1920, 1280
    w1, h1 = 1920, 1280
    # set up window for second display with fullscreen
    win1 = Toplevel(bg="white")
    win1.geometry(f"{w1}x{h1}")  # +{w0} <- this is the key, offset to the right by w0
    # win1.overrideredirect(True)
    projection = Projection(win1)
    # projection.draw_lines([[10, 20], [50, 60]], [[100, 400], [300, 500]])

    # instead of doing:
    # projection.draw_circles([[10, 10], [100, 100], [10, 20]], [10, 10, 10])

    # we could be doing this with the new dataclasses:

    def mock_game():
        lines = [[(10.0, 20.0), (50.0, 60.0)], [(100, 400), (300, 500)]]
        trajectories = [
            Trajectory(start_coordinate=Point(x=x1, y=y1), end_coordinate = Point(x=x2, y=y2))
            for (x1, y1), (x2,y2) in lines
        ]
        action = Action(stroke_angle=0.5, force='big force', predicted_trajectories = trajectories)
        points = [Point(x=10.0, y=10.0), Point(x=100.0, y=100.0), Point(x=10, y=20), Point(x=250, y=250)]
        ball_types = ['solid', 'cue', 'stripes', 'eight']
        balls = [BallState(point, type=ball_type) for point, ball_type in zip(points, ball_types)]
        table = TableState(balls)
        ball_radius = 10
        return action, table, ball_radius

    action, table, ball_radius = mock_game()
    projection.draw_circles([
        [ball.coordinate.x, ball.coordinate.y] for ball in table.ball_states],
        [ball_radius] * len(table.ball_states)
    )
    win1.mainloop()


main()