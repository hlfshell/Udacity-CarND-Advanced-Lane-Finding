import cv2
import pickle
import matplotlib.pyplot as plt


# Load up the camera-calibration pickle
with open("./camera_calibration.p", "rb") as filename:
    camera_calibration = pickle.load(filename)

# interest_area is the area of the road we're interested in - changeable in function, set here for ease of use
interest_area = [   [],
                    [],
                    [],
                    []
                ]

# This is a helper function to help me define the interest_area for test images.
def testInterestArea(area):
    pass

# given an image, undistory per camera calibration
def undistort(img, debug=False):
    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(camera_calibration["objPoints"], camera_calibration["imgPoints"], img.shape[::-1], None, None)
    # use calibration to undistort image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    if debug is True:
        plt.figure()
        plt.suptitle("Undistortion/Camera Correction")
        plt.imshow(undistorted)

    return undistorted


# perspective transform
# img - input image
# corners - list of tuples to make up the corner
# transform_area - the area we are transforming from
# to_area - the area that area should be warped to fit
# debug - whether or not to show debug information/image display on this step
def perspectiveTransform(img, corners, transform_area=None, to_area=None , debug=False):
    img_size = (img.shape[1], img.shape[0])

    # By default, the to_area is to make the warped perspective area the whole image size, done here:
    if to_area is None:
        to_area = [  [0, 0],
                     [img_size[0] - 1, 0],
                     [img_size[0] - 1, img_size[1] - 1],
                     [0, img_size[1] - 1]
                  ]

    # If no transform_area is provided, assume it's the interest_area by default
    if transform_area is None:
        global interest_area
        transform_area = interest_area

    # Convert destination points to points post perspective transformation
    transform_matrix = cv2.getPerspectiveTransform(transform_area, to_area)

    # Use the matrix to perform a perspective transformation
    warped = cv2.warpPerspective(img, transform_matrix, img_size)
    
    if debug is True:
        plt.figure()
        plt.suptitle("Perspective Transformation")
        plt.imshow(warped)

    return warped

#pipeline - accepts an image, returns ???
def pipeline(img, debug=False):

    # Undistort image

    # Color / gradient threshold

    # Perspective transform

    #  Detect lane lines

    # Calculate lane curvature 

    # Calculate distance from center for car

    pass