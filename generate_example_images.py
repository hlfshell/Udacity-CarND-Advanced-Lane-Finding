from pipeline import pipeline
import cv2
from glob import glob

images_folder = "./test_images"
output_folder = "./output_images"

images = glob(images_folder + "/*.jpg")

for filename in images:
    # read each image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, _, _, curve = pipeline(img)

    cv2.putText(result,"Curvature is {:0.2f}".format(curve), (10,650) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_folder + "/" + filename.split("/")[-1], result)