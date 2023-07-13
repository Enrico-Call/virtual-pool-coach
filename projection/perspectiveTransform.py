import cv2

import numpy as np

#Get transform first as in
#f = np.array([[0,0], [2, 0], [2, 1], [0, 1]], dtype = 'float32')
#t = np.array([[90, 80], [1860, 190], [1870, 990], [110, 1000]], dtype = 'float32')
#T=cv2.getPerspectiveTransform(f, t)

def transform_point(point, transform_matrix):
    newpoint=np.append(point, np.array(1))
    newpoint=transform_matrix.dot(newpoint)
    newpoint /= newpoint[2]
    return newpoint[0:2]

if __name__ == '__main__':
    f = np.array([[0,1.11], [2.23, 1.11], [2.23, 0], [0, 0]], dtype = 'float32')
    t = np.array([[70, 80], [1833, 173], [1878, 994], [101, 1027]], dtype = 'float32')
    point = np.array([0.0, 0.0])
    T=cv2.getPerspectiveTransform(f, t)
    print(T)
    print(transform_point(point, T))
    

