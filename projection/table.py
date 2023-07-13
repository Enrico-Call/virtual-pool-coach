import numpy as np
import cv2
import PIL.ImageTk, PIL.Image
import tkinter

class Table:
    def __init__(self, canvas, window_width, window_height, corners=None):
        self.width = window_width
        self.height = window_height
        self.canvas = canvas
        self.center = np.array([window_width / 2, window_height / 2])
        if corners is not None:
            print(f"Drawing calibration based on previously save points {corners}")
            self.corners = corners
        else:
            self.corners = self.init_corners(corners)
        self.current_calibration_index = 0
        self.pixel_delta = 10

    def draw(self):
        self.canvas.create_polygon(list(self.corners[0]), list(self.corners[1]),
                                                    list(self.corners[2]), list(self.corners[3]),
                                                    list(self.corners[0]), fill="white")
        current_corner_top_left = self.corners[self.current_calibration_index] - 10
        current_corner_bottom_right = self.corners[self.current_calibration_index] + 10
        self.canvas.create_oval(list(current_corner_top_left), list(current_corner_bottom_right), width=6, fill="white")
        self.canvas.pack(fill="both", expand=True)

    def draw_last_frame(self, frame, transform_matrix):
        frame = cv2.warpPerspective(frame, transform_matrix, (self.width, self.height))
        image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img)) 
        self.canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

    def move(self, event):
        key_pressed = event.keysym
        dx, dy = 0, 0
        # increase speed with which we move table
        if key_pressed == "e":
            self.pixel_delta += 1
        # decrease table movement speed
        if key_pressed == "q":
            self.pixel_delta = 0 if self.pixel_delta < 0 else self.pixel_delta - 1
        if key_pressed == "Left":
            self.corners[self.current_calibration_index][0] -= self.pixel_delta
        if key_pressed == "Right":
            self.corners[self.current_calibration_index][0] += self.pixel_delta
        if key_pressed == "Up":
            self.corners[self.current_calibration_index][1] -= self.pixel_delta
        if key_pressed == "Down":
            self.corners[self.current_calibration_index][1] += self.pixel_delta
        if key_pressed == "d":
            self.current_calibration_index += 1
            self.current_calibration_index %= 4
        if key_pressed == "a":
            self.current_calibration_index -= 1
            self.current_calibration_index %= 4


    def rotate(self):
        angle =  self.rotation_angle
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        self.top_left -= self.center
        self.bottom_right -= self.center
        self.top_right -= self.center
        self.bottom_left -= self.center
        self.top_left = np.dot(R, self.top_left)
        self.bottom_right = np.dot(R, self.bottom_right)
        self.top_right = np.dot(R, self.top_right)
        self.bottom_left = np.dot(R, self.bottom_left)

        self.top_left += self.center
        self.bottom_right += self.center
        self.top_right += self.center
        self.bottom_left += self.center

    def init_corners(self, corners):
        initial_width = 1600
        initial_height = 800
        top_left = self.center - np.array([initial_width / 2, initial_height / 2])
        bottom_right = self.center + np.array([initial_width / 2, initial_height / 2])
        top_right = self.center + np.array([initial_width / 2, -initial_height / 2])
        bottom_left = self.center + np.array([-initial_width / 2, initial_height / 2])
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

