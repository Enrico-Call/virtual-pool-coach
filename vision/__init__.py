import matplotlib.pyplot
from matplotlib import pyplot as plt

from game_model import TableState, BallState, Point
from typing import Any, Optional, Set

from .ball_classifier import BallTypeClassifier
from .ball_classifier.model import IMAGE_TRANSFORM
from .detect_balls import detect_balls
import cv2 as cv
import numpy as np
import math
import sys

MOCK_BALLS = {
    BallState(Point(x=1.1, y=1.2), type="stripes"),
    BallState(Point(x=2.1, y=2.2), type="cue"),
    BallState(Point(x=3.1, y=3.2), type="eight"),
    BallState(Point(x=4.1, y=4.2), type="solid")
}
MOCK_RADIUS = 15


def detect_table_state(frame: Optional[Any] = None, detector: Optional[Any] = None) -> TableState:
    # Mock implementation:
    if detector is None:
        params = cv.SimpleBlobDetector_Params()

        params.filterByColor = False
        params.blobColor = 1

        # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 200
        # params.maxArea = 500

        # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 0.2
        # params.maxCircularity = 3.4028234663852886e+38 # infinity.

        # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 1e-5
        # params.maxConvexity = 3.4028234663852886e+38

        # Filter by Inertia
        params.filterByInertia = False  # a second way to find round blobs.
        # params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat
        # params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

        # Change thresholds
        params.minThreshold = 0  # from where to start filtering the image
        params.maxThreshold = 255.0  # where to end filtering the image
        params.thresholdStep = 5  # steps to go through
        params.minDistBetweenBlobs = 1.0  # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution!
        params.minRepeatability = 2  # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.
        detector = cv.SimpleBlobDetector_create(params)

    if frame is None:
        print("frame is none")
        balls_states = MOCK_BALLS
    else:
        print("frame is not none")
        keypoints = detect_balls(frame, detector)
        print(keypoints)

        # im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
        #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        src_width, src_height = 700, 500
        src_ltop = [0, 0]
        src_lbot = [0, src_height]
        src_rtop = [src_width, 0]
        src_rbot = [src_width, src_height]
        source_coordinates = np.asarray(
            [src_ltop, src_lbot, src_rtop, src_rbot])  # pixel coordinates, changes by the day :D

        target_width, target_height = 2.24, 1.11  # in meters
        target_lbot = [0, 0]
        target_rtop = [target_width, target_height]
        target_ltop = [0, target_height]
        target_rbot = [target_width, 0]
        target_coordinates = [target_ltop, target_lbot, target_rtop, target_rbot]

        src = np.float32(source_coordinates)
        dst = np.float32(target_coordinates)
        perspective_transform = cv.getPerspectiveTransform(src, dst)
        center_balls = np.float32(np.array([[[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints]]))

        center_balls_in_meters = cv.perspectiveTransform(center_balls, perspective_transform)[0]

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        balls_states = get_ball_states_ml(frame, center_balls_px=center_balls[0],
                                          center_balls_in_meters=center_balls_in_meters)

        if all(bs.type != 'cue' for bs in balls_states):
            balls_states = []

    return TableState(ball_states=balls_states)


def get_ball_states(frame, center_balls_px, center_balls_in_meters) -> Set[BallState]:
    types = [detect_ball_types(frame, center) for center in center_balls_px]
    types[0] = 'cue'
    types[1] = 'eight'

    balls_states = {BallState(Point(x=x, y=y), type=type) for (x, y), type in zip(center_balls_in_meters, types)}
    return balls_states


BALL_TYPE_CLASSIFIER = BallTypeClassifier()


def get_ball_states_ml(frame, center_balls_px, center_balls_in_meters) -> Set[BallState]:
    type_probabilities = []
    for ball_position in center_balls_px:
        ball_crop = get_ball_crop(frame, ball_position, 16)
        ball_crop_tensor = IMAGE_TRANSFORM(ball_crop)
        if ball_crop.shape != (32, 32, 3):
            raise ValueError('Ooooopsy!')
        probabilities = BALL_TYPE_CLASSIFIER.predict_probabilities(ball_crop_tensor)
        type_probabilities.append(probabilities)

    max_p_cue = 0
    argmax_cue = -1
    for i, probs in enumerate(type_probabilities):
        p_cue = probs['cue']
        if p_cue > max_p_cue:
            max_p_cue = p_cue
            argmax_cue = i

    max_p_eight = -1
    argmax_eight = -1
    for i, probs in enumerate(type_probabilities):
        # prevent classifyin a ball as both cue and eight
        if i == argmax_cue:
            continue

        p_eight = probs['eight']
        if p_eight > max_p_eight:
            max_p_eight = p_eight
            argmax_eight = i

    types = [
        'stripes' if ball_p_distr['stripes'] > ball_p_distr['solid'] else 'solid'
        for ball_p_distr in type_probabilities
    ]
    types[argmax_cue] = 'cue'
    types[argmax_eight] = 'eight'

    balls_states = {BallState(Point(x=x, y=y), type=type) for (x, y), type in zip(center_balls_in_meters, types)}
    return balls_states


def get_ball_radius(x_position, y_position):
    '''Calculate radius from ball point'''

    return math.sqrt(math.pow(MOCK_RADIUS - x_position, 2) + math.pow(MOCK_RADIUS - y_position, 2))


def get_ball_crop(frame, position, half_width):
    frame_max_y, frame_max_x, _ = frame.shape

    x_start = int(position[0]) - half_width
    x_end = int(position[0]) + half_width
    if x_start < 0:
        delta = -x_start
        x_start = 0
        x_end += delta

    if x_end > frame_max_x:
        delta = x_end - frame_max_x
        x_start -= delta
        x_end = frame_max_x

    y_start = int(position[1]) - half_width
    y_end = int(position[1]) + half_width
    if y_start < 0:
        delta = -y_start
        y_start = 0
        y_end += delta

    if y_end > frame_max_y:
        delta = y_end - frame_max_y
        y_start -= delta
        y_end = frame_max_y

    return frame[y_start:y_end, x_start:x_end]


def get_ball_pixels(frame, position):
    '''Responsible for returning an array of pixels that represent the circle'''

    ball_pixels = []
    ball_frame = get_ball_crop(frame, position, half_width=MOCK_RADIUS).copy()

    for x_position, _ in enumerate(ball_frame[0:-1]):
        for y_position, _ in enumerate(ball_frame[0:-1]):
            if get_ball_radius(x_position, y_position) < MOCK_RADIUS:
                ball_pixels.append(ball_frame[x_position][y_position])

    return ball_pixels


def get_white_count(ball_pixels):
    '''Finding the number of white pixels within the ball pixels'''

    white_count = 0

    for pixel in ball_pixels:
        is_r_valid = 170 <= pixel[0] <= 255
        is_g_valid = 170 <= pixel[1] <= 255
        is_b_valid = 170 <= pixel[2] <= 255

        if is_r_valid and is_g_valid and is_b_valid:
            white_count += 1

    return white_count


def get_black_count(ball_pixels):
    '''Finding the number of black pixels within the ball pixels'''

    black_count = 0

    for pixel in ball_pixels:
        is_r_valid = 0 <= pixel[0] <= 80
        is_g_valid = 0 <= pixel[1] <= 80
        is_b_valid = 0 <= pixel[2] <= 80

        if is_r_valid and is_g_valid and is_b_valid:
            black_count += 1

    return black_count


def is_solid_ball(white_count, black_count):
    '''Checking whether the pixel count is a solid ball'''

    return 0 <= white_count <= 50 and 0 <= black_count <= 50


def is_striped_ball(white_count, black_count):
    '''Checking whether the pixel count is a striped ball'''

    return 26 <= white_count <= 309 and 0 <= black_count <= 99


def is_black_ball(white_count, black_count):
    '''Checking whether the pixel count is a black ball'''

    return 100 <= black_count <= 675


def is_white_ball(white_count):
    '''Checking whether the pixel count is a white ball'''

    return 310 <= white_count <= 675


def detect_ball_types(frame: np.array, keypoint) -> str:
    """Detect ball types (white, black, solid, striped).
    
    Args:
        frame:
            np.array in RGB format with the frame.
        balls:
            List of BallState objects with center ball location.
            
    Returns:
        Ball type as string.
    """

    ball_pixels = get_ball_pixels(frame, keypoint)
    white_pixels = get_white_count(ball_pixels)
    black_pixels = get_black_count(ball_pixels)

    if is_white_ball(white_pixels):
        return 'cue'
    elif is_black_ball(white_pixels, black_pixels):
        return 'eight'
    elif is_solid_ball(white_pixels, black_pixels):
        return 'solid'
    else:
        return 'stripes'
