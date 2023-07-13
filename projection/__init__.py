from pathlib import Path
import cv2
from game_model import Action, Point, Trajectory, TableState, BallState
from tkinter import Tk, Canvas, Toplevel, Button, Label
from .table import Table
from typing import List, Tuple
from functools import partial
import numpy as np
import pickle
import sys
from .camera import Camera

CONFIG_DIR = Path(__file__).parent.parent / 'config'
PROJECTION_CALIBRATION_FILE = CONFIG_DIR / 'calibration-projected-corners.pickle'
PROJECTION_TRANSFORMATION_MATRIX = CONFIG_DIR / 'calibration-projection-transformation-matrix.pickle'


class Projector():
    def start(self):
        # specify resolutions of both windows, w0 is computer window, w1 is projector
        self.w0, self.h0 = 1920, 1280
        self.w1, self.h1 = 1920, 1280
        self.win0 = Tk()
        self.win0.title("Virtual Pool Coach")
        self.gui_width = 500
        self.gui_height = 400
        self.win0.geometry(f"{self.gui_width}x{self.gui_height}")
        # set up window for second display with fullscreen
        self.win1 = Toplevel(bg="black")
        self.win1.geometry('%dx%d+%d+%d' % (self.w1,
                                            self.h1,
                                            self.w0,
                                            0))
        self.win1.overrideredirect(True)
        self.projected_corners = None
        self.projector_canvas = Canvas(self.win1, bg="black")
        self.expected_input = None
        self.waiting = False
        self.response = None
        self.mock_game = False
        self.init_gui()
        self.calibrate_projector()
        self.current_action = None
        self.current_table_state = None
        self.last_frame = None
        self.draw_last_frame = False

    def clear(self, canvas: Canvas) -> None:
        # Stop showing any actions
        canvas.delete("all")

    def update(self, draw_calibration_table=False):
        self.clear(self.projector_canvas)
        self.clear(self.gui_canvas)
        if draw_calibration_table:
            self.table.draw()
        elif self.draw_last_frame and self.last_frame is not None:
            self.table.draw_frame(self.last_frame, self.projection_transformation_matrix)
        self.draw()
        self.win1.update()
        self.win0.update()

    def draw(self):
        self.show_action(self.current_action)
        self.show_table_state(self.current_table_state)
        self.draw_gui()

    def stop(self) -> None:
        # Stop the tkinter UI
        self.win0.destroy()
        self.win1.destroy()

    def show_action(self, action: Action) -> None:
        if not action: return
        self.current_action = action
        trajectories = action.predicted_trajectories
        for trajectory in trajectories:
            self.draw_line(trajectory.start_coordinate, trajectory.end_coordinate, action.force)

    def show_table_state(self, table_state: TableState):
        if not table_state: return
        self.current_table_state = table_state
        ball_coords = []
        ball_types = []
        for ball in table_state.ball_states:
            ball_coords.append(ball.coordinate)
            ball_types.append(ball.type)
        self.draw_balls(ball_coords, ball_types)

    def draw_line(self, start_coord: Point, end_coord: Point, force: str = None, show_arrow: bool = True) -> None:
        if show_arrow:
            arrow = "last"
        else:
            arrow = "none"
        # color = "white"
        # width = 3
        # if force:
        #     color = self.force_to_line_features[force]["color"]
        #     width = self.force_to_line_features[force]["width"]
        start_coord = np.array([start_coord.x, start_coord.y], dtype="float32")
        end_coord = np.array([end_coord.x, end_coord.y], dtype="float32")
        transformed_start_coord = [int(x) for x in self.transform_point_to_pixels(start_coord)]
        transformed_end_coord = [int(x) for x in self.transform_point_to_pixels(end_coord)]
        self.projector_canvas.create_line(transformed_start_coord,
                                          transformed_end_coord,
                                          fill="yellow",
                                          width=4,
                                          arrow=arrow)
        self.projector_canvas.pack(fill="both", expand=True)

    def draw_balls(self, coords: List[Point], ball_types) -> None:
        type_to_style = {"stripes": {"fill": "yellow", "dash": 1},
                         "solid": {"fill": "green"},
                         "cue": {"fill": "white"},
                         "eight": {"fill": "white", "dash": 1}}
        for center_coord, ball_type in zip(coords, ball_types):
            style = type_to_style[ball_type]
            center_coord = np.array([center_coord.x, center_coord.y], dtype="float32")
            transformed_center_coord = [int(x) for x in
                                        self.transform_point_to_pixels(center_coord)]
            coord1, coord2 = self.get_circle_coords(transformed_center_coord, 40)
            self.projector_canvas.create_oval(coord1, coord2, width=10, **style)
            coord1, coord2 = self.get_circle_coords(transformed_center_coord, 5)
            self.projector_canvas.create_oval(coord1, coord2, width=5, fill=style["fill"])
        self.projector_canvas.pack(fill="both", expand=True)

    def get_circle_coords(self, center_coord: List, radius: int) -> Tuple:
        x0, y0 = [coord - radius for coord in center_coord]
        x1, y1 = [coord + radius for coord in center_coord]
        return [x0, y0], [x1, y1]


    def change_label_text(self, label: Label, text: str) -> None:
        """ Changes the text of a given label"""
        label.config(text=text)
        # self.update()

    def draw_gui(self):
        if self.expected_input == "select player":
            self.change_label_text(self.info_label, "Select what player is taking a shot")
            self.stripes_player_button.grid(column=1, columnspan=2, row=2,
                                            padx=self.padx, pady=self.pady, sticky='nsew')
            self.solid_player_button.grid(column=3, columnspan=2, row=2,
                                          padx=self.padx, pady=self.pady, sticky='nsew')
        if self.expected_input == "next turn":
            self.change_label_text(self.info_label, "Press to show the possible shots for the next turn")
            self.next_turn_button.grid(column=2, columnspan=2, row=2,
                                       padx=self.padx, pady=self.pady, sticky='nsew')
        if self.expected_input == "game end":
            self.change_label_text(self.info_label, "Has the game ended?")
            self.game_end_button.grid(columnspan=2, column=1, row=2,
                                      padx=self.padx, pady=self.pady, sticky='nsew')
            self.no_end_button.grid(columnspan=2, column=3, row=2,
                                      padx=self.padx, pady=self.pady, sticky='nsew')
        if self.expected_input == "decide action":
            self.change_label_text(self.info_label, "Do you want to try this shot or look for an alternative shot?")
            self.previous_action_button.grid(column=1, columnspan=2, row=2,
                                             padx=self.padx, pady=self.pady, sticky='nsew')
            self.alternative_action_button.grid(column=3, columnspan=2, row=2,
                                                padx=self.padx, pady=self.pady, sticky='nsew')
            self.done_with_action_button.grid(column=2, columnspan=2, row=3,
                                              padx=self.padx, pady=self.pady, sticky='nsew')

        self.info_label.grid(columnspan=self.grid_columns, row=0, padx=self.padx, pady=self.pady, sticky="nsew")
        self.error_label.grid(columnspan=self.grid_columns, row=1, padx=self.padx, pady=self.pady, sticky="nsew")
        # Footer stuff
        self.recalibrate_button.grid(column=0, columnspan=2, row=self.grid_rows-1, padx=self.padx, pady=self.pady, sticky="nsew")
        self.quit_button.grid(column=4,  columnspan=2, row=self.grid_rows-1, padx=self.padx, pady=self.pady, sticky="nsew")
        self.mock_game_button.grid(column=2, columnspan=2, row=self.grid_rows-1, padx=self.padx, pady=self.pady, sticky="nsew")
        self.gui_canvas.pack()

    def init_gui(self):
        # Initialize the canvas grid configuration
        self.padx = 5
        self.pady = 5
        self.grid_rows = 5
        self.grid_columns = 6
        self.text_color = "white"
        self.background_color = "grey10"
        self.button_color = "grey30"
        self.font = ("Helvetica", "9", "bold")
        self.label_font = ("Helvetica", '12', "bold")
        self.win0.config(bg=self.background_color)
        self.max_col_width = self.gui_width/self.grid_columns
        self.max_row_height = self.gui_height/self.grid_rows
        self.gui_canvas = Canvas(self.win0, bg=self.background_color)
        self.gui_canvas.grid(sticky="nsew")
        for i in range(self.grid_columns):
            self.gui_canvas.grid_columnconfigure(i, minsize=self.max_col_width, weight=0)
        for i in range(self.grid_rows):
            self.gui_canvas.grid_rowconfigure(i,  minsize=self.max_row_height, weight=0)
        self.create_buttons()

    def create_buttons(self):
        self.info_label = Label(self.gui_canvas, text="Welcome To VPC",
                                font=self.label_font,
                                bg=self.background_color, fg=self.text_color)
        self.error_label = Label(self.gui_canvas, text="",
                                 wraplength=self.gui_width - 10,
                                 font=self.font,
                                 bg=self.background_color, fg="red")
        self.recalibrate_button = Button(self.gui_canvas, text="Calibrate projector",
                                         wraplength=self.max_col_width-5,
                                         font=self.font,
                                         bg=self.button_color,
                                         fg=self.text_color,
                                         command=partial(self.calibrate_projector, True))
        self.quit_button = Button(self.gui_canvas, text="Quit",
                                  wraplength=self.max_col_width - 5,
                                  font=self.font,
                                  bg=self.button_color,
                                  fg=self.text_color,
                                  command=self.quit)
        self.next_turn_button = Button(self.gui_canvas, text="Next turn",
                                       wraplength=self.max_col_width - 5,
                                       font=self.font,
                                       bg=self.button_color,
                                       fg=self.text_color,
                                       command=self.next_turn)
        self.game_end_button = Button(self.gui_canvas, text="Game ended",
                                      wraplength=self.max_col_width - 5,
                                      font=self.font,
                                      bg=self.button_color,
                                      fg=self.text_color,
                                      command=self.game_end)
        self.no_end_button = Button(self.gui_canvas, text="Game didn't end",
                                      wraplength=self.max_col_width - 5,
                                      font=self.font,
                                      bg=self.button_color,
                                      fg=self.text_color,
                                      command=self.no_end)
        self.solid_player_button = Button(self.gui_canvas, text="Solid player",
                                          wraplength=self.max_col_width - 5,
                                          font=self.font,
                                          bg=self.button_color,
                                          fg=self.text_color,
                                          command=self.select_solid_player)
        self.stripes_player_button = Button(self.gui_canvas, text="Striped player",
                                            wraplength=self.max_col_width - 5,
                                            font=self.font,
                                            bg=self.button_color,
                                            fg=self.text_color,
                                            command=self.select_stripes_player)
        self.alternative_action_button = Button(self.gui_canvas, text="Next Action",
                                                wraplength=self.max_col_width - 5,
                                                font=self.font,
                                                bg=self.button_color,
                                                fg=self.text_color,
                                                command=self.alternative_action)
        self.previous_action_button = Button(self.gui_canvas, text="Previous Action",
                                             wraplength=self.max_col_width - 5,
                                             font=self.font,
                                             bg=self.button_color,
                                             fg=self.text_color,
                                             command=self.previous_action)
        self.done_with_action_button = Button(self.gui_canvas, text="Done with Action",
                                              wraplength=self.max_col_width - 5,
                                              font=self.font,
                                              bg=self.button_color,
                                              fg=self.text_color,
                                              command=self.done_with_action)
        self.mock_game_button = Button(self.gui_canvas, text="Start mock game",
                                       wraplength=self.max_col_width - 5,
                                       font=self.font,
                                       bg=self.button_color,
                                       fg=self.text_color,
                                       command=self.start_mock_game)


    def start_mock_game(self):
        from .mock_game import get_all_actions, TABLE_STATES
        self.mock_game = True
        table_state_index = 0
        action_index = 0
        self.win1.bind("<Escape>", self.stop_mock_game)
        while self.mock_game:
            self.current_table_state = TABLE_STATES[table_state_index]
            all_actions = get_all_actions(self.current_table_state)
            while True:
                if action_index < len(all_actions):
                    self.current_action = all_actions[action_index]
                else:
                    print("All actions on this table state loaded")
                self.update()
        self.win1.unbind("<Escape>")


    def stop_mock_game(self):
        self.mock_game = False

    def quit(self):
        print("EXIT")
        self.stop()
        sys.exit()

    def done_with_action(self):
        if self.expected_input == "decide action":
            self.response = "D"
            self.expected_input = None
            self.previous_action_button.grid_remove()
            self.alternative_action_button.grid_remove()
            self.done_with_action_button.grid_remove()
            self.waiting = False

    def previous_action(self):
        if self.expected_input == "decide action":
            self.response = "S"
            self.expected_input = None
            self.previous_action_button.grid_remove()
            self.alternative_action_button.grid_remove()
            self.done_with_action_button.grid_remove()
            self.waiting = False

    def alternative_action(self):
        if self.expected_input == "decide action":
            self.response = "A"
            self.expected_input = None
            self.previous_action_button.grid_remove()
            self.alternative_action_button.grid_remove()
            self.done_with_action_button.grid_remove()
            self.waiting = False

    def select_stripes_player(self):
        if self.expected_input == "select player":
            self.response = "stripes"
            self.expected_input = None
            self.stripes_player_button.grid_remove()
            self.solid_player_button.grid_remove()
            self.waiting = False

    def select_solid_player(self):
        if self.expected_input == "select player":
            self.response = "solid"
            self.expected_input = None
            self.stripes_player_button.grid_remove()
            self.solid_player_button.grid_remove()
            self.waiting = False


    def next_turn(self):
        if self.expected_input == "next turn":
            self.response = None
            self.expected_input = None
            self.next_turn_button.grid_remove()
            self.waiting = False

    def game_end(self):
        if self.expected_input == "game end":
            self.response = "Y"
            self.expected_input = None
            self.game_end_button.grid_remove()
            self.no_end_button.grid_remove()
            self.waiting = False

    def no_end(self):
        if self.expected_input == "game end":
            self.response = "N"
            self.expected_input = None
            self.no_end_button.grid_remove()
            self.game_end_button.grid_remove()
            self.waiting = False

    def wait_for_input(self, expected_input):
        self.waiting = True
        self.expected_input = expected_input
        while self.waiting:
            self.update()
        return self.response


    def calibrate_projector(self, recalibrate=False) -> None:
        if self.check_existing_calibration() and not recalibrate:
            print(f"Using existing calibration with corner points {self.projected_corners}")
            return
        # Focus on projector window to make arrow keys work immediately
        self.win1.focus_force()
        self.table = Table(self.projector_canvas, self.w1, self.h1, self.projected_corners)
        self.table.draw()
        self.win1.bind("<Key>", self.table.move)
        self.win1.bind("<Return>", self.stop_calibration)
        self.win1.bind("<Escape>", self.escape_calibration)
        self.calibrating = True
        print("Starting calibration process, press enter when the corners align")
        # Custom mainloop for calibration
        while self.calibrating:
            self.update(True)
        self.win1.unbind("<Key>")
        self.win1.unbind("<Return")
        self.win1.unbind("<Escape>")


    def stop_calibration(self, event) -> None:
        """
            Stop the calibration process and save the calibration points and matrix in a pickle file
        """
        self.projected_corners = self.table.corners
        self.create_projection_transformation_matrix()
        self.calibrating = False
        print(f"Found corner pockets at pixel coordinates {self.projected_corners}")
        print("Calibration has ended")
        print("Saving current calibration")
        # Save the current calibrations
        with PROJECTION_CALIBRATION_FILE.open(mode='wb') as f:
            pickle.dump(self.projected_corners, f)
        with PROJECTION_TRANSFORMATION_MATRIX.open(mode='wb') as f:
            pickle.dump(self.projection_transformation_matrix, f)
        self.update()

    def escape_calibration(self, event):
        """ Stop the calibration process without saving this calibration"""
        self.calibrating = False
        self.update()

    def check_existing_calibration(self):
        if PROJECTION_CALIBRATION_FILE.exists() and PROJECTION_TRANSFORMATION_MATRIX.exists():
            with PROJECTION_CALIBRATION_FILE.open(mode='rb') as f:
                self.projected_corners = pickle.load(f)
            with PROJECTION_TRANSFORMATION_MATRIX.open(mode='rb') as f:
                self.projection_transformation_matrix = pickle.load(f)
            return True
        return False

    def transform_point_to_pixels(self, point):
        newpoint = np.append(point, np.array(1))
        newpoint = self.projection_transformation_matrix.dot(newpoint)
        newpoint /= newpoint[2]
        return newpoint[0:2]

    def create_projection_transformation_matrix(self):
        table_dimensions = np.array([[0.0, 1.11], [2.24, 1.11], [2.24, 0.0], [0.0, 0.0]], dtype='float32')
        self.projection_transformation_matrix = cv2.getPerspectiveTransform(table_dimensions,
                                                                            self.projected_corners)

