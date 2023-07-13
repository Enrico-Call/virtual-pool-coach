
import cv2
import numpy as np


class Camera:
    def __init__(self, feed=None, index=0):
        """
            If feed is not None then it should be a path to a video file.
            Index is the index of the camera on the computer,
            if an integrated camera exists,
            external camera should be index 1,
            and integrated index 0.
        """# You should replace these 3 lines with the output in calibration step using 
        self.camera_dimensions=(1280, 720)
        self.K=np.array([[651.7132994936426, 0.0, 634.3076605698704], [0.0, 649.0018389101608, 355.71976632973406], [0.0, 0.0, 1.0]])
        self.D=np.array([[-0.00792507864820157], [0.025826539191918532], [-0.050614508901507795], [0.03196607439042139]])


        self.feed = feed
        self.index = index
        if self.feed is not None:
            self.camera = cv2.VideoCapture(feed)
        else:
            self.camera = cv2.VideoCapture(self.index)

    def _undistort(self, img):
        h,w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.camera_dimensions, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    
    def isOpened(self):
        return self.camera.isOpened()

    def read(self):
        if self.feed is not None:
            return self.camera.read()
        else:
            ret, frame = self.camera.read()
            return ret, self._undistort(frame)

    def __del__(self):
        if self.feed is not None:
            self.camera.release()
