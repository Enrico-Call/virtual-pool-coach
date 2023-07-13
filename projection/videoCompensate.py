# import the opencv library
import cv2
import numpy as np
from defisheye import Defisheye
import itertools
import time
# You should replace these 3 lines with the output in calibration step using 
DIM=(1280, 720)
K=np.array([[651.7132994936426, 0.0, 634.3076605698704], [0.0, 649.0018389101608, 355.71976632973406], [0.0, 0.0, 1.0]])
D=np.array([[-0.00792507864820157], [0.025826539191918532], [-0.050614508901507795], [0.03196607439042139]])
def undistort(img):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


# define a video capture object
vid = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_writer = cv2.VideoWriter('recordings/balls/coolshot.mp4', fourcc, 20.0, (1280, 720))
#time.sleep(60) 
for i in range(1):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if ret==True:
        frame = undistort(frame)
        out_writer.write(frame)
        #cv2.imwrite("recordings/balls/white-ball-1.png", frame)
        cv2.imshow('frame', frame)
        #cv2.imwrite(f'recordings/balls/blank.jpg', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
