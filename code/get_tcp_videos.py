from geometric_helpers import *
from find_tcp import *
import cv2

######################### CALIBRATION LOOP #########################
def getRawTCPImages(leftDirectory, rightDirectory, jsonDirectory, targetFrames, exposureLeft, exposureRight, is2MMTip):

    #### Store the initial points in which the TCP was detected ####
    frames = 0
    lcam_points = []
    rcam_points = []
    initial_points_found = False

    ### Set detection parameters ###
    # Canny Threshold
    cannyThreshMin = 25
    cannyThreshMax = 100
    # Line Accumulator Matrix
    lineAccMin = 100
    lineAccMax = 300
    # Circle Accumulator Matrix
    circleAccMin = 15
    circleAccMax = 300
    # Minimum circle radius
    minRadiusMin = 50
    if (is2MMTip):
        minRadiusMin = 30
    minRadiusMax = 80
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
    minDistMin = 20
    minDistMax = 100
    # Maximum distance between edges      
    maxDistMin = 2000
    maxDistMax = 2500
    # Error for tip and axis alignment
    errorMin = 20
    errorMax = 100

    ## Window 1 - Left Camera Feed ##
    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 700, 700)
    win1_name = "Window 1"

    cv2.createTrackbar('Canny Threshold', win1_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win1_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win1_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win1_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win1_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win1_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win1_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win1_name, minDistMin, minDistMax, lambda a: None)
    # # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win1_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win1_name, errorMin, errorMax, lambda a: None)

    ## Window 2  - Right Camera Feed ##
    cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 2", 700, 700)
    win2_name = "Window 2"

    cv2.createTrackbar('Canny Threshold', win2_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win2_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win2_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win2_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win2_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win2_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win2_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win2_name, minDistMin, minDistMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win2_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win2_name, errorMin, errorMax, lambda a: None)

    # Open cameras
    (cam_array, frame_counts, converter,
     cameraMatrix746, distCoeffs746, 
     cameraMatrix745, distCoeffs745) = openCamerasAndCalibrationFiles(exposureLeft, exposureRight)

    # Detect the tool center point
    cam_array.StartGrabbing()
    while (frames < targetFrames or not initial_points_found):
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

        # Snap Images
        (image_left, image_right) = snap_tcp_images(cam_array, frame_counts, converter, 
                                                    cameraMatrix746, distCoeffs746, 
                                                    cameraMatrix745, distCoeffs745)
        
        ### Write the image to the corresponding directory if TCP found ###
        if (initial_points_found):
            cv2.imwrite(f"{leftDirectory}/{frames:05d}.jpg", image_left)
            cv2.imwrite(f"{rightDirectory}/{frames:05d}.jpg", image_right)
            frames += 1

            # Motivation..
            if (frames == targetFrames / 3):
                print("One third of the way there!")
            elif (frames == targetFrames / 2):
                print("One half of the way there!")
            elif (frames == 2 * targetFrames / 3):
                print("Two thirds of the way there!")

        # Otherwise, detect TCP 
        if (not initial_points_found):
            (tcp_left, central_axis_pointl, color_image_left, left_edges) = detectTCP(image_left, cannyThresholdL, 
                                line_threshL, circ_threshL, 
                                minLineLengthL, maxLineGapL, 
                                minDistBtwnEdgesL, maxDistBtwnEdgesL, 
                                minRadiusL, maxRadiusL, 
                                dispToleranceL, 0.85)
            
            (tcp_right, central_axis_pointr, color_image_right, right_edges) = detectTCP(image_right, cannyThresholdR,
                                line_threshR, circ_threshR,
                                minLineLengthR, maxLineGapR,
                                minDistBtwnEdgesR, maxDistBtwnEdgesR,
                                minRadiusR, maxRadiusR,
                                dispToleranceR, 0.85)
            image_left = color_image_left
            image_right = color_image_right
            
            if (tcp_left is not None and tcp_right is not None and
                central_axis_pointl is not None and central_axis_pointr is not None):

                ### Save the detected point for SAM2 ###
                cv2.imwrite(f"{leftDirectory}/{frames:05d}.jpg", image_left)
                cv2.imwrite(f"{rightDirectory}/{frames:05d}.jpg", image_right)
                frames += 1

                lcam_points.append([0, tcp_left[0][0], tcp_left[1][0]])
                lcam_points.append([0, central_axis_pointl[0], central_axis_pointl[1]])
                rcam_points.append([0, tcp_right[0][0], tcp_right[1][0]])
                rcam_points.append([0, central_axis_pointr[0], central_axis_pointr[1]])

                cv2.circle(color_image_left, (int(tcp_left[0][0]), int(tcp_left[1][0])), 5, (0, 255, 0), 4)   # center
                cv2.circle(color_image_left, (int(central_axis_pointl[0]), int(central_axis_pointl[1])),    # shaft
                            5, (255, 255, 0), 4)
                cv2.circle(color_image_right, (int(tcp_right[0][0]), int(tcp_right[1][0])), 5, (0, 255, 0), 4) # center
                cv2.circle(color_image_right, (int(central_axis_pointr[0]), int(central_axis_pointr[1])),    # shaft
                            5, (255, 255, 0), 4)
                
                cv2.imshow(win1_name, color_image_left)
                cv2.imshow(win2_name, color_image_right)
                cv2.waitKey(2)  # visually verify that the points reside within the shaft - o/w, start over

                print("Initial point found for SAM2!")
                initial_points_found = True

                
            
        # Display images
        cv2.imshow(win1_name, image_left)
        cv2.imshow(win2_name, image_right)

       # q escape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    
    ### Write initial points to json file ###
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cam_array.StopGrabbing()
    cam_array.Close()

    clean_left = [[int(f), int(x), int(y)] for f, x, y in lcam_points]
    clean_right = [[int(f), int(x), int(y)] for f, x, y in rcam_points]
    
    points_dict = {
        "left_points": clean_left,
        "right_points": clean_right
    }

    # Convert everything to lists in one go
    clean_data = {k: v for k, v in points_dict.items()}

    with open(jsonDirectory, 'w') as f:
        json.dump(clean_data, f, indent=4)