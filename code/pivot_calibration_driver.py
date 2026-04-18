import pypylon.pylon as py
from geometric_helpers import *
from charuco_calibration import *
from find_tcp import *
import numpy as np
import cv2
import math
import json


# TODO: refine hough circles to be more accomodating (currently trying hough_circles_alt)
# Note: top-down lighting from a little far away worked (on 3/29)
# TODO: try Cutie for spatial and temporal stability

######################### DETECTION LOOP #########################


def pivotCalibration():
    (cam_array, frame_counts, converter,
     cameraMatrix746, distCoeffs746, 
     cameraMatrix745, distCoeffs745, F) = openCamerasAndCalibrationFiles()
    
    ### Detection Parameters ###
    # Canny Threshold
    cannyThreshMin = 30
    cannyThreshMax = 100
    # Line Accumulator Matrix
    lineAccMin = 100
    lineAccMax = 300
    # Circle Accumulator Matrix
    circleAccMin = 20
    circleAccMax = 300
    # Minimum circle radius
    minRadiusMin = 30
    minRadiusMax = 50
    # Maximum circle radius
    maxRadiusMin = 100
    maxRadiusMax = 200
    # Minimum line length
    minLineMin = 10
    minLineMax = 100
    # Maximum line gap
    maxLineMin = 50
    maxLineMax = 250
    # Minimum distance between edges
    minDistMin = 200
    minDistMax = 400
    # Maximum distance between edges      
    maxDistMin = 800
    maxDistMax = 3000
    # Error for tip and axis alignment
    errorMin = 10
    errorMax = 50

    ## Window 1 - Left Camera Feed ##
    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 800, 650)
    win_name = "Window 1"

    cv2.createTrackbar('Canny Threshold', win_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win_name, minDistMin, minDistMax, lambda a: None)
    # # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win_name, errorMin, errorMax, lambda a: None)

    ## Window 2  - Right Camera Feed ##
    cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 2", 800, 650)
    win_name = "Window 2"

    cv2.createTrackbar('Canny Threshold', win_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win_name, minDistMin, minDistMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win_name, errorMin, errorMax, lambda a: None)

    # Detect the tool center point
    poses = []
    while (len(poses) < 20):
        # 1. Move robot
            # TODO: fill this out

        # 2. Snap Images
        (image_left, image_right) = snap_tcp_images(cam_array, frame_counts, converter, 
                                                    cameraMatrix746, distCoeffs746, 
                                                    cameraMatrix745, distCoeffs745)
        
        # Get left camera trackbar positions
        cannyThresholdL = cv2.getTrackbarPos('Canny Threshold', "Window 1")    
        line_threshL = cv2.getTrackbarPos('Line Accumulator Threshold', "Window 1")
        circ_threshL = cv2.getTrackbarPos('Circle Accumulator Threshold', "Window 1")
        minLineLengthL = cv2.getTrackbarPos('Minimum Line Length', "Window 1")
        maxLineGapL = cv2.getTrackbarPos('Maximum Line Gap', "Window 1")
        minDistBtwnEdgesL = cv2.getTrackbarPos('Minimum Distance Between Edges', "Window 1")
        maxDistBtwnEdgesL = cv2.getTrackbarPos('Maximum Distance Between Edges', "Window 1")
        minRadiusL = cv2.getTrackbarPos('Minimum Radius', "Window 1")
        maxRadiusL = cv2.getTrackbarPos('Maximum Radius', "Window 1")
        dispToleranceL = cv2.getTrackbarPos('Maximum Error for Tip and Axis Alignment', "Window 1")

        # Get right camera trackbar positions
        cannyThresholdR = cv2.getTrackbarPos('Canny Threshold', "Window 2")    
        line_threshR = cv2.getTrackbarPos('Line Accumulator Threshold', "Window 2")
        circ_threshR = cv2.getTrackbarPos('Circle Accumulator Threshold', "Window 2")
        minLineLengthR = cv2.getTrackbarPos('Minimum Line Length', "Window 2")
        maxLineGapR = cv2.getTrackbarPos('Maximum Line Gap', "Window 2")
        minDistBtwnEdgesR = cv2.getTrackbarPos('Minimum Distance Between Edges', "Window 2")
        maxDistBtwnEdgesR = cv2.getTrackbarPos('Maximum Distance Between Edges', "Window 2")
        minRadiusR = cv2.getTrackbarPos('Minimum Radius', "Window 2")
        maxRadiusR = cv2.getTrackbarPos('Maximum Radius', "Window 2")
        dispToleranceR = cv2.getTrackbarPos('Maximum Error for Tip and Axis Alignment', "Window 2")

    
    # # Calculate its 3D position relative to the camera
    # worldCoordinates = []
    # if (points745 is not None):
    #     worldPoint = grab3DPoints("./calibration_data/external_parameters.json", points746, points745)
    #     print(f"World Point: {worldPoint}")
    #     # worldCoordinates.append(worldPoint)

    # return worldCoordinates


# Runs the main loop until the TCP is detected by both cameras (loop broken with 'q' key)
def runDetection(win_name, cam_array, frame_counts, converter, cameraMatrix745, distCoeffs745, cameraMatrix746, distCoeffs746, F):
               
    

    (image_left, image_right) = snap_tcp_images(cam_array, frame_counts, converter, 
                                                cameraMatrix746, distCoeffs746, 
                                                cameraMatrix745, distCoeffs745)
                

    # if (cam_id == 0):
    #     cv2.imshow('Window 1', color_image)
        
    # elif (cam_id == 1):
    #     cv2.imshow('Window 2', color_image)
    
    # # check if q button has been pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
                    
    # cam_array.StopGrabbing()
    # cam_array.Close()
    # return (points745, points746)


######################### Driver #########################
