import numpy as np
import cv2
import matplotlib.pyplot as plt

def obtain_warped_table_view(img,left_top=(250,148), left_bottom=(240,550), right_top=(1085,145), right_bottom=(1087,560), width=500, height=700, visualise=False):
    """
        this function takes 4 points on an image that should align with the corners of the canvas of size (w,h)
        and warps the image accordingly such that only the inside of the pool table is detectable.

        the output should be passed to the object detection part

        NOTE: the corner stones [left_top, left_bottom, right_top, right_bottom]
        have to be determined manually and depends on the output of the camera.

        see https://theailearner.com/tag/cv2-warpperspective/ for more information on how cv2.warpPerspective() works

    :param img: input image of the table that is already corrected for the fisheye lens distortion
    :return inside_table: image containing only the "inside" of the pool table.
    """
    pts1 = np.float32([list(left_top), list(left_bottom), list(right_top), list(right_bottom)])

    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height] ])

    matrix = cv2.getPerspectiveTransform(pts1,pts2) # getting perspective by 4 points of each image
    transformed = cv2.warpPerspective(img, matrix, (width,height)) # warps perpective to new image

    out=cv2.transpose(transformed)
    inside_table = out
    # inside_table=cv2.flip(out, flipCode=1)

    # imgplot = plt.imshow(inside_table)
    #plt.savefig("inside_table.jpg")
    if visualise:
        plt.show()
        plt.scatter(left_top[0], left_top[1], s=20, c='red', marker='o')
        plt.scatter(left_bottom[0], left_bottom[1], s=20, c='red', marker='o')
        plt.scatter(right_bottom[0], right_bottom[1], s=20, c='red', marker='o')
        plt.scatter(right_top[0], right_top[1], s=20, c='red', marker='o')
        imgplot2 = plt.imshow(img)
        plt.show()
    return inside_table

# width = 500
# height = 700
# left_top = [253,150]
# left_bottom = [245,557]
# right_top = [1090,147]
# right_bottom = [1097,575]
# img_path = "table_img_2501.png" # you can find this img in onebox
# img = cv2.imread(img_path)
# visualise = True # True will also plot the coordinates on the original image
# inside_table = obtain_warped_table_view(img,
#                          left_top,
#                          left_bottom,
#                          right_top,
#                          right_bottom,
#                          width,
#                          height,
#                          visualise)
# plt.imshow(inside_table)
# plt.show()
