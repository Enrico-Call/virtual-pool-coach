# +
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import mediapy
from etils import ecolab
from rich.jupyter import print
ecolab.auto_plot_array()
# -
VIDEO_PATH = "/mnt/c/Users/hwx1192260/Downloads/clusters.mp4"

# Read the video
# Convert to an array of images
# Run the detection of the balls

capture = cv.VideoCapture(VIDEO_PATH)


# +
i = 0

import sys
sys.path.append("../")

from vision.warp_table import obtain_warped_table_view
from vision.detect_balls import detect_balls
from scipy.ndimage import binary_fill_holes

params = cv.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 0

# Filter by Area.
params.filterByArea = False
params.minArea = 20
params.maxArea = 600

# Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 0.2
# params.maxCircularity = 3.4028234663852886e+38 # infinity.

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 1e-5
# params.maxConvexity = 3.4028234663852886e+38

# Filter by Inertia
params.filterByInertia = False # a second way to find round blobs.
params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat
params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

# Change thresholds
params.minThreshold = 10 # from where to start filtering the image
params.maxThreshold = 255 # where to end filtering the image
params.thresholdStep = 10 # steps to go through
params.minDistBetweenBlobs = 1 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution!
params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs),

gamma = .5

def gamma_correction(frame, gamma=1):
    inv = 1. / gamma
    table = (((np.arange(256) / 255.) * inv) * 255.)
    return cv.LUT(frame, table).astype(np.uint8)
    

detector = cv.SimpleBlobDetector_create(params)

while capture.isOpened():
    _, frame = capture.read() # frame is is BGR color space
    # mediapy.show_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if frame is not None:
        unwrapped = obtain_warped_table_view(frame)
        corrected = gamma_correction(unwrapped, gamma=gamma)
        # keypoints = detect_balls(unwrapped, detector)
        hsv_frame = cv.cvtColor(corrected, cv.COLOR_BGR2HSV)
        gray = hsv_frame[..., 1]
        _, th = cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        opened = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=2)
        filled = binary_fill_holes(~opened)
        filled = (~filled).astype(np.uint8) * 255
        keypoints = detector.detect(filled)
        print(f"{len(keypoints)} balls detected")
        im_with_keypoints = cv.drawKeypoints(unwrapped, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        mediapy.show_images([cv.cvtColor(frame, cv.COLOR_BGR2RGB), cv.cvtColor(unwrapped, cv.COLOR_BGR2RGB), cv.cvtColor(corrected, cv.COLOR_BGR2RGB), gray, th, opened, filled, cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2RGB)], columns=2)
        
        if i == 0:
            break # temporary break
    else:
        break
    i = i+1


capture.release()

# +
from vision import detect_table_state

table_state = detect_table_state(unwrapped, detector)
print(table_state)


# +
import random
import dataclasses
from strategy.strategy import visualize_strategy
from game_model import GameState

cue_ball = table_state.ball_states[0]
cue_ball = dataclasses.replace(cue_ball, type="cue")
eight_ball = table_state.ball_states[1]
eight_ball = dataclasses.replace(eight_ball, type="eight")

 

balls = [dataclasses.replace(b, type="stripes" if random.random() > .5 else "solid") for b in table_state.ball_states[2:]]
balls.extend([cue_ball, eight_ball])

table_state2 = dataclasses.replace(table_state, ball_states=balls)
game_state = GameState(table_state=table_state2, turn="stripes")
print(game_state)
# -

visualize_strategy(game_state, [])
