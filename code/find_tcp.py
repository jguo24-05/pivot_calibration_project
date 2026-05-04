from geometric_helpers import *
from get_tcp_images import *
import numpy as np
import cv2

######################### DETECTION LOGIC #########################
# Calculate world coordinates based on projection matrices 
# points must be formatted as np.array([[center_x], [center_y]], dtype=np.float32)
# def calculateWorldPoint(json_path, projPoints1, projPoints2):
#      with open(json_path, 'r') as file:
#         json_data = json.load(file)
#         projMatr1 = np.array(json_data["745_projection_mtx"])
#         projMatr2 = np.array(json_data["746_projection_mtx"])
#         homogeneous = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)

#         print(homogeneous[3])

#         points_3d = homogeneous[:3] / homogeneous[3]
#         return points_3d.flatten()                      ## TODO: is this correct?
     
def calculateWorldPoint(json_path, projPoints1, projPoints2):
    with open(json_path, 'r') as file:
        data = json.load(file)
        
        # Load Rectified Projection Matrices (from your JSON)
        P1 = np.array(data["745_projection_mtx"])
        P2 = np.array(data["746_projection_mtx"])
        
        # You need these from your calibration to "clean" the points
        K1 = np.array(data["745_intrinsic_mtx"]) # Ensure these exist in JSON
        D1 = np.array(data["745_dist_coeffs"])
        R1 = np.array(data["745_rect_transform"]) # From stereoRectify
        
        K2 = np.array(data["746_intrinsic_mtx"])
        D2 = np.array(data["746_dist_coeffs"])
        R2 = np.array(data["746_rect_transform"])

        # Step 1: Map raw pixels to Rectified Plane
        # Use P1 and P2 here so the points end up in the coordinate system the P matrices expect
        points1_rect = cv2.undistortPoints(projPoints1, K1, D1, R=R1, P=P1)
        points2_rect = cv2.undistortPoints(projPoints2, K2, D2, R=R2, P=P2)

        # Step 2: Triangulate the "clean" points
        homogeneous = cv2.triangulatePoints(P1, P2, points1_rect, points2_rect)
        
        # Step 3: Normalize
        world_coords = homogeneous[:3] / homogeneous[3]
        return world_coords.flatten()


################# Attempts to find the TCP in the given image #################
def detectTCP(color_image, cannyThreshold, 
                line_thresh, circ_thresh, 
                minLineLength, maxLineGap, 
                minDistBtwnEdges, maxDistBtwnEdges, 
                minRadius, maxRadius, 
                dispTolerance, parallelTolerance):
    
    color_image = color_image.copy()

    # 1. Preprocessing (Grayscale + Gaussian Blur)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Blur is crucial to reduce noise/specular highlights on metal surfaces
    # blurred = cv2.GaussianBlur(gray, (9, 9), 15)
    blurred = cv2.medianBlur(gray, 15)

    # 2. Detect Drill Tip Handle Edges with Hough Probabilistic Transform 
    cannyMinThreshold = cannyThreshold / 3
    edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold)
    
    lineTuple = detectLines(edges,
                            color_image, 
                            line_thresh, 
                            minLineLength, 
                            maxLineGap, 
                            minDistBtwnEdges, 
                            maxDistBtwnEdges, 
                            parallelTolerance)   
    
    if (lineTuple is None):
        cv2.putText(color_image, "The tool shaft was not detected this frame", (70, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    if (lineTuple is not None):
        line1 = lineTuple[0]
        line2 = lineTuple[1]
        ax1, ay1, ax2, ay2 = line1[0]
        cx1, cy1, cx2, cy2 = line2[0]
        
        topx1 = ax1
        topy1 = ay1
        botx1 = ax2
        boty1 = ay2
        if (ay2 > ay1):
            topx1 = ax2
            topy1 = ay2
            botx1 = ax1
            boty1 = ay1

        topx2 = cx1
        topy2 = cy1
        botx2 = cx2
        boty2 = cy2
        if (cy2 > cy1):
            topx2 = cx2
            topy2 = cy2
            botx2 = cx1
            boty2 = cy1
        
        bx1 = int((topx1+topx2)/2.0)
        by1 = int((topy1+topy2)/2.0)
        bx2 = int((botx1+botx2)/2.0)
        by2 = int((boty1+boty2)/2.0)
        centralAxis = ((bx1, by1), (bx2, by2))
        centralAxisMidpoint = ((bx1+bx2)/2, (by1+by2)/2)

        cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
        cv2.line(color_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
        cv2.line(color_image, centralAxis[0], centralAxis[1], (125, 125, 0), 5)

        # 3. Detect Drill Tip (Ball/Sphere) Using Hough Circles if Edges Were Detected
        accumulatorRes = 1     
        minDist = 2000;             

        circle = detectCircles(blurred, 
                                accumulatorRes, 
                                minDist, 
                                cannyThreshold, 
                                circ_thresh, 
                                minRadius, 
                                maxRadius, 
                                centralAxis, 
                                dispTolerance)
        
        ## 4. TCP Discovered! ##
        if (circle is None):
            cv2.putText(color_image, "The tool shaft was discovered this frame, but not the drill tip", (70, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        if not (circle is None):
            center = circle[0]
            radius = circle[1]
            center_x = center[0]
            center_y = center[1]

            tcp = np.array([[center_x], [center_y]], dtype=np.float32)
                
            ## Draw the TCP in the frame ##
            # micronsOverPixels = 1000/radius
            
            cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(color_image, (center_x, center_y), 2, (0, 0, 255), 3)
            # cv2.putText(color_image, f"Microns per pixel: {micronsOverPixels: .2f}", 
            #             (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
            
            return (tcp, centralAxisMidpoint, color_image, edges)
    
    return (None, None, color_image, edges)


################# Attempts to find the TCP in the given image #################
def detectTCPAlt(color_image, cannyThreshold, 
                line_thresh, circ_thresh, 
                minLineLength, maxLineGap, 
                minDistBtwnEdges, maxDistBtwnEdges, 
                minRadius, maxRadius, 
                dispTolerance, parallelTolerance):
    
    color_image = color_image.copy()

    # 1. Preprocessing (Grayscale + Gaussian Blur)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Blur is crucial to reduce noise/specular highlights on metal surfaces
    # blurred = cv2.GaussianBlur(gray, (9, 9), 15)
    blurred = cv2.medianBlur(gray, 15)
    cannyMinThreshold = cannyThreshold
    edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold)

    # 2. Detect Circles
    accumulatorRes = 1     
    minDist = 2000
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                                dp = accumulatorRes, minDist = minDist, param1 = cannyThreshold,   # for hough_gradient_alt 
                                param2 = circ_thresh, minRadius = minRadius, 
                                maxRadius = maxRadius)
    
    linePair = detectLines(edges,
                            color_image, 
                            line_thresh, 
                            minLineLength, 
                            maxLineGap, 
                            minDistBtwnEdges, 
                            maxDistBtwnEdges, 
                            parallelTolerance)   
    
    if (circles is None or linePair is None):
        return (None, None, color_image, edges)
    
    circles = np.round(circles).astype("uint16")
    for i in circles[0,:]:
        center = (i[0], i[1])
        radius = i[2]
        center_x = center[0]
        center_y = center[1]

        cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(color_image, (center_x, center_y), 2, (0, 0, 255), 3)

        line1 = linePair[0]
        line2 = linePair[1]

        ax1, ay1, ax2, ay2 = line1[0]
        cx1, cy1, cx2, cy2 = line2[0]
        
        bx1 = int((ax1+cx1)/2.0)
        by1 = int((ay1+cy1)/2.0)
        bx2 = int((ax2+cx2)/2.0)
        by2 = int((ay2+cy2)/2.0)
        centralAxis = ((bx1, by1), (bx2, by2))
        centralAxisMidpoint = ((bx1+bx2)/2, (by1+by2)/2)

        if (pointOnLine(center, centralAxis[0], centralAxis[1], dispTolerance)):
            cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
            cv2.line(color_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
            cv2.line(color_image, centralAxis[0], centralAxis[1], (125, 125, 0), 5)
            tcp = np.array([[center_x], [center_y]], dtype=np.float32)
                
            # cv2.putText(color_image, f"Microns per pixel: {micronsOverPixels: .2f}", 
            #             (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
            
            return (tcp, centralAxisMidpoint, color_image, edges)

    return (None, None, color_image, edges)