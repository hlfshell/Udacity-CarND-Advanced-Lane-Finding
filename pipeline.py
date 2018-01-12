import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import randint


# Load up the camera-calibration pickle
with open("./camera_calibration.p", "rb") as filename:
    camera_calibration = pickle.load(filename)

# interest_area is the area of the road we're interested in - changeable in function, set here for ease of use
interest_area = np.float32([
                    [480, 456],
                    [900, 456],
                    [1200, 670],
                    [140, 670]
                ])

# This is a helper function to help me define the interest_area for test images.
def testInterestArea(img, area=None):
    global interest_area

    if area is None:
        area = interest_area

    cv2.polylines(img, np.int32([area]), True, color=(255, 0, 0))
    plt.figure()
    plt.suptitle("Interest Area")
    plt.imshow(img)
    plt.show()

def grabTestImage(imgNum=None):
    if imgNum is None:
        imgNum = randint(1, 6)

    img = cv2.imread("./test_images/test" + str(imgNum) + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# given an image, undistory per camera calibration
def undistort(img, debug=False):
    image_size = (img.shape[1], img.shape[0])

    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(camera_calibration["objPoints"], camera_calibration["imgPoints"], image_size, None, None)

    # use calibration to undistort image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    if debug is True:
        plt.figure()
        plt.suptitle("Undistortion/Camera Correction")
        plt.imshow(undistorted)
        plt.show()

    return undistorted

# Color / gradient threholding
def colorGradientThreshold(img, debug=False):
    pass


# perspective transform
# img - input image
# transform_area - the area we are transforming from
# to_area - the area that area should be warped to fit
# debug - whether or not to show debug information/image display on this step
def perspectiveTransform(img, transform_area=None, to_area=None , debug=False):
    img_size = (img.shape[1], img.shape[0])

    # By default, the to_area is to make the warped perspective area the whole image size, done here:
    if to_area is None:
        to_area = np.float32([ 
                     [0, 0],
                     [img_size[0] - 1, 0],
                     [img_size[0] - 1, img_size[1] - 1],
                     [0, img_size[1] - 1]
                  ])

    # If no transform_area is provided, assume it's the interest_area by default
    if transform_area is None:
        global interest_area
        transform_area = interest_area

    print(to_area, transform_area)

    # Convert destination points to points post perspective transformation
    transform_matrix = cv2.getPerspectiveTransform(transform_area, to_area)

    # Use the matrix to perform a perspective transformation
    warped = cv2.warpPerspective(img, transform_matrix, img_size)
    
    if debug is True:
        plt.figure()
        plt.suptitle("Perspective Transformation")
        plt.imshow(warped)
        plt.show()

    return warped

#pipeline - accepts an image, returns ???
def pipeline(img, debug=False):

    # Undistort image
    undistorted = undistort(img, debug)

    # Color / gradient threshold


    # Perspective transform
    transformed = perspectiveTransform(img, debug=debug)

    #  Detect lane lines

    # Calculate lane curvature 

    # Calculate distance from center for car

    pass