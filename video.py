from moviepy.editor import VideoFileClip
from pipeline import pipeline
import cv2

previousLines = []

highway_line_width = 3.7

def handleFrame(image):
    
    highlightedLane, left_lane_polynomial, right_lane_polynomial, averageCurveRadius, metersOffCenter, highwayWidth = pipeline(image, previousLanes=previousLines)

    line = { "img": highlightedLane, "left_lane_polynomial": left_lane_polynomial, "right_lane_polynomial": right_lane_polynomial, "curve": averageCurveRadius, "offcenter": metersOffCenter, "lanewidth": highwayWidth }

    previousLines.append(line)
    # # set up first run through
    # if previousLine is None:
    #     previousLine = line
    # elif abs( (line["offcenter"] - previousLine["offcenter"]) / line["offcenter"] ) > 0.075:

    #     if line["offcenter"] > previousLine["offcenter"]:
    #         line["offcenter"] = 1.05 * previousLine["offcenter"]
    #     else:
    #         line["offcenter"] = 0.95 * previousLine["offcenter"]

    #     if line["curve"] > previousLine["curve"]:
    #         line["curve"] = 1.05 * previousLine["curve"]
    #     else:
    #         line["curve"] = 0.95 * previousLine["curve"]

    #     line["img"] = previousLine["img"]

    # #Likewise, if the lane highwidth is 10 % different, reject the current frame anyway for the previous one
    # if abs((highway_line_width - line["lanewidth"]) / highway_line_width) >= 0.1:
    #     line = previousLine

    # previousLine = line

    # Add curvature text to the image
    cv2.putText(highlightedLane,"Curvature is {:0.2f}".format(line["curve"]), (10,600) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(highlightedLane,"Distance from center is {:0.2f} meters".format(line["offcenter"]), (10,650) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    return highlightedLane    

clip1 = VideoFileClip("./project_video.mp4")
output_clip = clip1.fl_image(handleFrame)
output_clip.write_videofile("./output_images/project_video_output.mp4", audio=False)
