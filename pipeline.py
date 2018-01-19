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
                    [480, 500],
                    [850, 500],
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
    inverse_transform_matrix = cv2.getPerspectiveTransform(to_area, transform_area)

    # Use the matrix to perform a perspective transformation
    warped = cv2.warpPerspective(img, transform_matrix, img_size)
    
    if debug is True:
        plt.figure()
        plt.suptitle("Perspective Transformation")
        plt.imshow(warped)
        plt.show()

    return warped, transform_matrix, inverse_transform_matrix

#detectLaneLines
# img - input image (overhead threshold)
# debug - whether or not to show debug info
# returns leftLinePoints, rightLinePoints
def detectLaneLines(img, debug=False):
    histogram = np.sum( img[ int(img.shape[0]/2):,:], axis=0)

    if debug is True:
        # Create an output image to draw on - make it 3 channel from our single channel
        # and multiply by 255 to set the 1's to 255
        out_img = np.dstack( (img, img, img) ) * 255

    # Define our image midpoint (via histogram shape, since we're working with that)
    midpoint = np.int(histogram.shape[0]/2)

    # Left peak base
    leftx_base = np.argmax(histogram[:midpoint])
    # Right peak base
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # How many sliding windows/convolutions do we do vertically?
    nWindows = 9 
    
    # Height of each window is calculated off of that
    window_height = np.int(img.shape[0] / nWindows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions that are updated on each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set width of windows w/ margin
    margin = 100

    # Set the minimum number of pixels found to recenter window
    minpix = 50

    # Empty lists to hold each index of center point of left/right lanes
    left_lane_indexes = []
    right_lane_indexes = []

    # Go through window step by step
    for window in range(nWindows):
        
        window_y_low = img.shape[0] - (window + 1) * window_height
        window_y_high = img.shape[0] - window * window_height

        window_xleft_low = leftx_current - margin
        window_xleft_high = leftx_current + margin
        window_xright_low = rightx_current - margin
        window_xright_high = rightx_current + margin

        #Draw windows if debugging
        if debug is True:
            cv2.rectangle(out_img, (window_xleft_low, window_y_low), (window_xleft_high, window_y_high), color= (0,0,255))
            cv2.rectangle(out_img, (window_xright_low, window_y_low), (window_xright_high, window_y_high), color= (0,0,255))

        good_left_indexes = ( (nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high) ).nonzero()[0]
        good_right_indexes = ( (nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_xright_low) & (nonzerox < window_xright_high) ).nonzero()[0]

        left_lane_indexes.append(good_left_indexes)
        right_lane_indexes.append(good_right_indexes)

        # If we detect below a certain # of pixels, recenter window
        if len(good_left_indexes) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_indexes]))
        if len(good_right_indexes) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_indexes]))

    left_lane_indexes = np.concatenate(left_lane_indexes)
    right_lane_indexes = np.concatenate(right_lane_indexes)

    leftx = nonzerox[left_lane_indexes]
    lefty = nonzeroy[left_lane_indexes]
    rightx = nonzerox[right_lane_indexes]
    righty = nonzeroy[right_lane_indexes]

    # Fit a polynomial to each
    left_lane_polynomial = np.polyfit(lefty, leftx, 2)
    right_lane_polynomial = np.polyfit(righty, rightx, 2)

    # If debugging, draw lane lines
    if debug is True:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_lane_polynomial[0]*ploty** 2 + left_lane_polynomial[1]*ploty + left_lane_polynomial[2]
        right_fitx = right_lane_polynomial[0]*ploty**2 + right_lane_polynomial[1]*ploty + right_lane_polynomial[2]

        out_img[nonzeroy[left_lane_indexes], nonzerox[left_lane_indexes]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indexes], nonzerox[right_lane_indexes]] = [0, 0, 255]

        plt.figure()
        plt.suptitle("Extrapolated Lane Lines")
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Calculate the curvature

    # First, how many meters per pixel (given via udacity - camera specific)
    meters_per_pix_y = 30/720
    meters_per_pix_x = 3.7/700

    # How wide is a highway lane? (3.7m)
    highway_width = 3.7

    y = img.shape[0] - 1

    left_lane_meters_polynomial = np.polyfit(lefty * meters_per_pix_y, leftx * meters_per_pix_x, 2)
    right_lane_meters_polynomial = np.polyfit(righty * meters_per_pix_y, rightx * meters_per_pix_x, 2)

    leftCurveRadius =  ((1 + (2*left_lane_meters_polynomial[0]*y*meters_per_pix_y + left_lane_meters_polynomial[1])**2)**1.5) / np.absolute(2*left_lane_meters_polynomial[0])
    rightCurveRadius = ((1 + (2*right_lane_meters_polynomial[0]*y*meters_per_pix_y + right_lane_meters_polynomial[1])**2)**1.5) / np.absolute(2*right_lane_meters_polynomial[0])

    # Average curve should be the difference
    averageCurveRadius = (leftCurveRadius + rightCurveRadius) / 2

    # Calculate how far off from center the car is, assuming the camera is centered
    left_lane_bottom = (left_lane_polynomial[0]*y**2) + (left_lane_polynomial[1]*y) + (left_lane_polynomial[2])
    right_lane_bottom = (right_lane_polynomial[0]*y**2) + (right_lane_polynomial[1]*y) + (right_lane_polynomial[2])

    center = int(img.shape[1] / 2) # halfway point of image is our dead center

    calculated_center = (right_lane_bottom + left_lane_bottom) / 2
    widthInPixels = right_lane_bottom - left_lane_bottom
    metersPerPixelForLane = highway_width / widthInPixels

    metersOffCenter = metersPerPixelForLane * (calculated_center - center)

    return left_lane_polynomial, right_lane_polynomial, leftCurveRadius, rightCurveRadius, averageCurveRadius, lefty, leftx, righty, rightx, metersOffCenter

# drawLane - draws in the lane via a polygon calculated from polynomials
# img - input
# left_lane - polynomial for lane curvature
# right_lane - polynomial for lane curvature
# returns - image with polygon drawn over lane
def drawLane(img, overhead, Minv, lefty, leftx, righty, rightx):

    #We draw the polynomial in the overhead clone first
    canvas = np.zeros_like(overhead).astype(np.uint8)
    canvas = np.dstack((canvas, canvas, canvas))

    drawnLane = cv2.fillPoly(canvas, np.int32([list(zip(leftx, lefty)) + list(reversed(list(zip(rightx, righty))))]), color=(0, 255, 0) )

    print(overhead.shape, canvas.shape)

    unwarpedLane = cv2.warpPerspective(canvas, Minv, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(img, 1, unwarpedLane, 0.3, 0)

    return result

def drawLane2(img, overhead, Minv, left_line, right_line):

    #We draw the polynomial in the overhead clone first
    canvas = np.zeros_like(overhead).astype(np.uint8)
    canvas = np.dstack((canvas, canvas, canvas))

    y_min = int(interest_area[0][1])
    y_max = int(interest_area[2][1])

    h, w = img.shape[:2]
    x = np.arange(w)

    left = lambda x: (left_line[0] * x**2) + (left_line[1] * x) + left_line[2]
    right = lambda x: (right_line[0] * x**2) + (right_line[1] * x) + right_line[2]

    for h in range(0, img.shape[:2][0]):
        leftPoint = left(h)
        rightPoint = right(h)
        
        for w in range(0, img.shape[:2][1]):
            if w >= leftPoint and w <= rightPoint:
                canvas[h][w] = (0, 255, 0)

    unwarpedLane = cv2.warpPerspective(canvas, Minv, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(img, 1, unwarpedLane, 0.3, 0)

    return result

#pipeline - accepts an image, returns ???
def pipeline(img, debug=False):

    # Undistort image
    undistorted = undistort(img, debug=debug)

    # Color and Sobel thresholds
    combinedThreshold = combinedThresholds(undistorted, debug=debug)

    # Perspective transform
    overheadThreshold, transform_matrix, inverse_transform_matrix = perspectiveTransform(combinedThreshold, debug=debug)

    #  Detect lane lines
    left_lane_polynomial, right_lane_polynomial, leftCurveRadius, rightCurveRadius, averageCurveRadius, lefty, leftx, righty, rightx, metersOffCenter = detectLaneLines(overheadThreshold, debug)

    # Draw onto image lane
    # highlightedLane = drawLane(img, overheadThreshold, inverse_transform_matrix, lefty, leftx, righty, rightx)

    highlightedLane = drawLane2(img, overheadThreshold, inverse_transform_matrix, left_lane_polynomial, right_lane_polynomial)

    return highlightedLane, leftCurveRadius, rightCurveRadius, averageCurveRadius, metersOffCenter