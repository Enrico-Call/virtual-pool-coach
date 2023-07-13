from typing import List

from game_model import Action, Turn, GameState, WIDTH, HEIGHT, LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP
from strategy import compute_next_move
from vision import detect_table_state
from projection import Projector, Camera
import cv2 as cv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from vision.warp_table import obtain_warped_table_view
import matplotlib.pyplot as plt


class Controller:
    projector: Projector
    feed: str

    def __init__(self, feed=None):
        self.projector = Projector()
        self.feed = feed
        self.camera = Camera(self.feed)

    def run(self):
        self.projector.start()
        while self.camera.isOpened():
            self.projector.update()
            self.projector.wait_for_input("next turn")
            self.projector.change_label_text(self.projector.error_label, "Opening camera")
            _, frame = self.camera.read() # frame is is BGR color space
            plt.imshow(frame)
            plt.show()
            unwrapped_frame = obtain_warped_table_view(frame, LEFT_TOP, LEFT_BOTTOM, RIGHT_TOP, RIGHT_BOTTOM, WIDTH, HEIGHT)
            plt.imshow(unwrapped_frame)
            plt.show()
            self.projector.last_frame = unwrapped_frame
            self.projector.change_label_text(self.projector.error_label, "Detecting game state")
            table_state = detect_table_state(unwrapped_frame)
            if table_state.is_game_over():
                response = self.projector.wait_for_input("game end")
                if response.upper() == 'Y':
                    break
                elif response.upper() == "N":
                    self.projector.change_label_text(self.projector.error_label, "Sorry our bad")
                    self.projector.update()
                    continue

            turn: Turn = None
            if table_state.has_ball_been_potted():
                response = self.projector.wait_for_input("select player")
                if response == 'solid':
                    turn = 'solid'
                elif response == 'stripes':
                    turn = 'stripes'

            game_state = GameState(
                table_state=table_state,
                turn=turn
            )
            self.projector.change_label_text(self.projector.error_label, "Game state has been detected")

            self.projector.show_table_state(game_state.table_state)
            actions: List[Action] = compute_next_move(game_state)

            cur_action_idx = 0

            if len(actions) == 0:
                self.projector.change_label_text(self.projector.error_label,
                                                 "We cannot detect any possible moves on the table, this could be an error, or the table is not setup. Let's try again")
                continue
            while True:
                action = actions[cur_action_idx]
                self.projector.show_action(action)

                action_command = self.projector.wait_for_input("decide action")
                if action_command.upper() == "D":
                    self.projector.current_action = None
                    self.projector.current_table_state = None
                    break

                if action_command.upper() == "A":
                    if cur_action_idx == len(actions) - 1:
                        cur_action_idx = 0
                    else:
                        cur_action_idx += 1
                elif action_command.upper() == "S":
                    if cur_action_idx == 0:
                        cur_action_idx = len(actions) - 1
                    else:
                        cur_action_idx -= 1

        self.projector.stop()
