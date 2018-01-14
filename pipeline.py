import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import randint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEFAULTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

color_thresholds = { "h": (15, 100), "s": (90, 255) }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
# img - the image being worked on
# thresholds - dict of { "h": tuple, "s": tuple } where each tuple is (low, high)
# returns hue_mask, saturation_mask, combined_mask
def colorThreshold(img, thresholds=None, debug=False):
    if thresholds is None:
        thresholds = color_thresholds

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]

    if debug is True:
        plt.figure()
        plt.suptitle("Hue mask")
        plt.imshow(h_channel)
        plt.show()
        plt.figure()
        plt.suptitle("Saturation mask")
        plt.imshow(s_channel)
        plt.show()

    binary_mask_h = np.zeros_like(h_channel)
    binary_mask_s = np.zeros_like(s_channel)
    binary_mask_combined = np.zeros_like(h_channel)

    binary_mask_h[ (h_channel > thresholds["h"][0]) & (h_channel <= thresholds["h"][1]) ] = 1
    binary_mask_s[ (s_channel > thresholds["s"][0]) & (s_channel <= thresholds["s"][1]) ] = 1
    binary_mask_combined[ (h_channel > thresholds["h"][0]) & (h_channel <= thresholds["h"][1]) & (s_channel > thresholds["s"][0]) & (s_channel <= thresholds["s"][1]) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Binary Mask of Hue Thresholds")
        plt.imshow(binary_mask_h)
        plt.show()

        plt.figure()
        plt.suptitle("Binary Mask of Saturation Thresholds")
        plt.imshow(binary_mask_s)
        plt.show()

        plt.figure()
        plt.suptitle("Binary Mask of Hue/Saturation Thresholds")
        plt.imshow(binary_mask_combined)
        plt.show()

    return binary_mask_h, binary_mask_s, binary_mask_combined

def sobelGradient(img, threshold=[20, 100], debug=True):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))

    if debug is True:
        plt.figure()
        plt.suptitle("Absolute Scaled Sobel X Gradient")
        plt.imshow(scaled_sobel)
        plt.show()

    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[ (scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1]) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Absolute Scaled Sobel Threshold")
        plt.imshow(sobel_binary, cmap="gray")
        plt.show()


    return scaled_sobel, sobel_binary

def combinedThresholds(img, debug=True):
    hue_mask, saturation_mask, combined_mask = colorThreshold(img, debug=debug)
    scaled_sobel, sobel_threshold = sobelGradient(img, debug=debug)

    combined = np.dstack(( np.zeros_like(scaled_sobel), scaled_sobel, combined_mask)) * 255

    print(combined)

    if debug is True:
        plt.figure()
        plt.suptitle("Combined Gradients and Thresholds")
        plt.imshow(combined)
        plt.show()

    return combined

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

    # Color threshold
    # colorThreshold

    #Gradient threshold

    # Perspective transform
    transformed = perspectiveTransform(img, debug=debug)

    #  Detect lane lines

    # Calculate lane curvature 

    # Calculate distance from center for car

    pass