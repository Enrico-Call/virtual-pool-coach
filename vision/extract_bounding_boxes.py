import cv2 as cv
from matplotlib import pyplot as plt

from vision import detect_table_state
from vision.warp_table import obtain_warped_table_view

width = 500
height = 700
left_top = [255,160]
left_bottom = [250,572]
right_top = [1090,147]
right_bottom = [1103,580]

r = 10
c = 5
frame = cv.imread("C:\\Users\\lwx1200269\\PycharmProjects\\vpcmain\\table_img_2501.png")
unwrapped_frame = obtain_warped_table_view(frame, left_top, left_bottom, right_top, right_bottom, width, height)
table_state = detect_table_state(unwrapped_frame, extract_data=True)

for ball in table_state.ball_states:
    x = ball.coordinate.x
    y = ball.coordinate.y
    half_width = r + c



    x_left = int(x) - half_width
    x_right = int(x) + half_width
    y_top = int(y) - half_width
    y_bottom = int(y) + half_width
    print(unwrapped_frame.shape)
    cropped_ball = unwrapped_frame[y_top:y_bottom,x_left:x_right]
    print(cropped_ball.shape)
    plt.imshow(cropped_ball)
    plt.show()
    print()


plt.imshow(unwrapped_frame)
plt.show()