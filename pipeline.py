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

def absoluteSobel(img, threshold={ "x": (40, 255), "y": (80, 255) }, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255 * abs_sobely/np.max(abs_sobely))

    sobel_binaryx = np.zeros_like(scaled_sobelx)
    sobel_binaryx[ (scaled_sobelx >= threshold["x"][0]) & (scaled_sobelx <= threshold["x"][1]) ] = 1

    sobel_binaryy = np.zeros_like(scaled_sobely)
    sobel_binaryy[ (scaled_sobely >= threshold["y"][0]) & (scaled_sobely <= threshold["y"][1]) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Absolute Scaled Sobel X Gradient")
        plt.imshow(scaled_sobelx)
        plt.show()

        plt.figure()
        plt.suptitle("Absolute Scaled Sobel X Threshold")
        plt.imshow(sobel_binaryx, cmap="gray")
        plt.show()

        plt.figure()
        plt.suptitle("Absolute Scaled Sobel Y Gradient")
        plt.imshow(scaled_sobely)
        plt.show()

        plt.figure()
        plt.suptitle("Absolute Scaled Sobel Y Threshold")
        plt.imshow(sobel_binaryy, cmap="gray")
        plt.show()


    return scaled_sobelx, sobel_binaryx, scaled_sobely, sobel_binaryy

def magnitudeSobel(img, threshold=(60, 255), debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)

    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    scaled_sobel = np.uint8(255*sobel/np.max(sobel))

    binary_mask = np.zeros_like(scaled_sobel)
    binary_mask[ (scaled_sobel > threshold[0]) & (scaled_sobel < threshold[1]) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Magnitude Sobel Threshold")
        plt.imshow(scaled_sobel, cmap="gray")
        plt.show()

        plt.figure()
        plt.suptitle("Magnitude Sobel Binary Mask")
        plt.imshow(binary_mask, cmap="gray")
        plt.show()

    return scaled_sobel, binary_mask

def directionalSobel(img, threshold=(0.85, 1.0), debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)

    directionalSobel = np.absolute(np.arctan(sobely/sobelx))
    directionalBinary = np.zeros_like(directionalSobel)
    directionalBinary[ (directionalSobel > threshold[0]) & (directionalSobel < threshold[1]) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Directional Sobel Mask")
        plt.imshow(directionalSobel)
        plt.show()
        
        plt.figure()
        plt.suptitle("Directional Sobel Binary Mask")
        plt.imshow(directionalBinary)
        plt.show()

    return directionalSobel, directionalBinary

def combinedSobel(img, debug=True):
    _, sobel_binaryx, _, sobel_binaryy = absoluteSobel(img, debug=debug)
    _, magnitude_binary = magnitudeSobel(img, debug=debug)
    # _, directional_binary = directionalSobel(img, debug=debug)

    combined = np.zeros_like(sobel_binaryx)
    # combined[ (sobel_binaryx == 1) | (sobel_binaryy == 1) | (magnitude_binary == 1) | (directional_binary == 1) ] = 1
    combined[ (sobel_binaryx == 1) | (sobel_binaryy == 1) | (magnitude_binary == 1) ] = 1

    if debug is True:
        plt.figure()
        plt.suptitle("Combined Sobel Gradients")
        plt.imshow(combined)
        plt.show()

    return combined

def combinedThresholds(img, debug=False):
    sobelMask = combinedSobel(img, debug=debug)
    hMask, sMask, combinedColorMask = colorThreshold(img, debug=debug)

    combined = np.zeros_like(sobelMask)
    combined[ (combinedColorMask > 0) | (sobelMask > 0) ]=1

    if debug is True:
        plt.figure()
        plt.suptitle("Combined Sobel and Color Thresholds")
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
    undistorted = undistort(img, debug=debug)

    # Color and Sobel thresholds
    combinedThreshold = combinedThresholds(img, debug=debug)

    # Perspective transform
    overheadThreshold = perspectiveTransform(combinedThreshold, debug=debug)

    plt.figure()
    plt.suptitle("Overhead threshold")
    plt.imshow(overheadThreshold)
    plt.show()

    #  Detect lane lines

    # Calculate lane curvature 

    # Calculate distance from center for car

    pass