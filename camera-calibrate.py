# This is a util file that will, given calibration images in the folder
# camera_cal, create a camera calibration for distortion/perspective
# changes and save it to a pickle file for use in other functions/files

import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import pickle

images_folder = "./camera_cal"
output_file = 'camera_calibration.p'

# Inside corners in X and y, respectively:
nX = 9
nY = 6

objPoints = [] # objPoints are predefined points where I want the corners to go
imgPoints = [] # imgPoints are detected chessboard corners 

#Create an array of object points
objPoint = np.zeros((nX * nY, 3), np.float32)
objPoint[:,:2] = np.mgrid[0:nX, 0:nY].T.reshape(-1, 2) # X, Y

# Grab images and iterate over them 
images = glob(images_folder + "/calibration*.jpg")

for filename in images:
    #Read each image in - remember, since it's opencv, it's BGR!
    img = cv2.imread(filename)

    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find corners
    ret, corners = cv2.findChessboardCorners(gray, (nX, nY), None)

    if ret == True:
        imgPoints.append(corners)
        objPoints.append(objPoint)

# Save the imgPoints and objPoints to a pickle file
toSave = { "imgPoints": imgPoints, "objPoints": objPoints }
with open(output_file, 'wb') as saveFile:
    pickle.dump(toSave, saveFile)