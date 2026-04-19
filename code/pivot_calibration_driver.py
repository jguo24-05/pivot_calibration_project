from geometric_helpers import *
from find_tcp import *
import cv2

######################### CALIBRATION LOOP #########################
def pivotCalibration():
    (cam_array, frame_counts, converter,
     cameraMatrix746, distCoeffs746, 
     cameraMatrix745, distCoeffs745) = openCamerasAndCalibrationFiles(200000, 200000)
    
    ### Detection Parameters ###
    # Canny Threshold
    cannyThreshMin = 20
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
    minDistMin = 250
    minDistMax = 400
    # Maximum distance between edges      
    maxDistMin = 400
    maxDistMax = 800
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

    # Counter for profiling instability
    calibration_attempts = 0
    successful_detections = 0

    # Detect the tool center point
    poses = []
    cam_array.StartGrabbing()
    while (len(poses) < 20):
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

        # 1. Move robot
        # TODO: integrate this with robot code, just a key press for now            
            
        # 2. Snap Images
        (image_left, image_right) = snap_tcp_images(cam_array, frame_counts, converter, 
                                                    cameraMatrix746, distCoeffs746, 
                                                    cameraMatrix745, distCoeffs745)
        # 3. Detect TCP
        (tcp_left, _) = detectTCP(image_left, cannyThresholdL, 
                            line_threshL, circ_threshL, 
                            minLineLengthL, maxLineGapL, 
                            minDistBtwnEdgesL, maxDistBtwnEdgesL, 
                            minRadiusL, maxRadiusL, 
                            dispToleranceL, 0.3)
        
        (tcp_right, _) = detectTCP(image_right, cannyThresholdR,
                            line_threshR, circ_threshR,
                            minLineLengthR, maxLineGapR,
                            minDistBtwnEdgesR, maxDistBtwnEdgesR,
                            minRadiusR, maxRadiusR,
                            dispToleranceR, 0.3)

        # 3. Triangulate the point to find the 3D coordinate relative to the camera center
        if (tcp_left is not None and tcp_right is not None):
            successful_detections += 1
            worldCoordinate = calculateWorldPoint("./calibration_data/external_parameters.json", tcp_left, tcp_right)
            x = worldCoordinate[0]
            y = worldCoordinate[1]
            z = worldCoordinate[2]
            cv2.putText(image_left, f"Position: {x:.2f}, {y:.2f}, {z:.2f}", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(image_right, f"Position: {x:.2f}, {y:.2f}, {z:.2f}", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            poses.append(worldCoordinate)
            # Write to a log
            with open("./calibration_log.txt", "a") as f:
                f.write(f"{worldCoordinate}\n")
                f.write(f"Calibration Attempts: {calibration_attempts}\n")
                f.write(f"Successful Detections: {successful_detections}\n")
                f.write("-" * 20 + "\n") # Optional separator line
            calibration_attempts = 0
            successful_detections = 0

        # 4. If a tcp was not detected, write a message to the frame
        if (tcp_left is None):
            cv2.putText(image_left, "TCP not found in this frame", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        if (tcp_right is None):
            cv2.putText(image_right, "TCP not found in this frame", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # 5. Display images
        cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window 1', 700, 700)
        cv2.imshow('Window 1', image_left)
        cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window 2', 700, 700)
        cv2.imshow('Window 2', image_right)
       # q escape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        calibration_attempts += 1

    cam_array.StopGrabbing()
    cam_array.Close()



######################### Driver #########################
pivotCalibration()
