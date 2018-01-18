from moviepy.editor import VideoFileClip
from pipeline import pipeline
import cv2

previousLine = None

def handleFrame(image):
    
    global previousLine
    
    highlightedLane, leftCurveRadius, rightCurveRadius, averageCurveRadius, metersOffCenter= pipeline(image)

    line = { "img": highlightedLane, "left": leftCurveRadius, "right": rightCurveRadius, "curve": averageCurveRadius, "offcenter": metersOffCenter }

    # set up first run through
    if previousLine is None:
        previousLine = line
    elif abs(line["curve"] - previousLine["curve"]) > 0.05 * previousLine["curve"]:
        #If the change is over 5%, ignore it, mark the chaneg to only 5%
        if line["curve"] > previousLine["curve"]:
            line["curve"] = 1.05 * previousLine["curve"]
        else:
            line["curve"] = 0.95 * previousLine["curve"]
        line["img"] = previousLine["img"]

    previousLine = line

    # Add curvature text to the image
    cv2.putText(highlightedLane,"Curvature is {:0.2f}".format(line["left"]), (10,600) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(highlightedLane,"Distance from center is {:0.2f}".format(line["offcenter"]), (10,650) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    return highlightedLane    

clip1 = VideoFileClip("./project_video.mp4")
output_clip = clip1.fl_image(handleFrame)
output_clip.write_videofile("./output_images/project_video_output.mp4", audio=False)
