from pipeline import undistort
import cv2
from glob import glob

images_folder = "./test_images"
output_folder = "./undistorted_images"

images = glob(images_folder + "/*.jpg")

for filename in images:
    # read each image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = undistort(img)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_folder + "/" + filename.split("/")[-1], result)