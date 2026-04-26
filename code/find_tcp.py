from geometric_helpers import *
from get_tcp_images import *
import numpy as np
import cv2

######################### DETECTION LOGIC #########################
# Calculate world coordinates based on projection matrices 
# points must be formatted as np.array([[center_x], [center_y]], dtype=np.float32)
def calculateWorldPoint(json_path, projPoints1, projPoints2):
     with open(json_path, 'r') as file:
        json_data = json.load(file)
        projMatr1 = np.array(json_data["745_projection_mtx"])
        projMatr2 = np.array(json_data["746_projection_mtx"])
        homogeneous = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)
        points_3d = homogeneous[:3] / homogeneous[3]
        return points_3d.flatten()                      ## TODO: is this correct?


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
    cannyMinThreshold = cannyThreshold
    edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold);
    
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
        
        bx1 = int((ax1+cx1)/2.0)
        by1 = int((ay1+cy1)/2.0)
        bx2 = int((ax2+cx2)/2.0)
        by2 = int((ay2+cy2)/2.0)
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
    minDist = 2000;  
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
        return (None, edges)
    
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

        if (pointOnLine(center, centralAxis[0], centralAxis[1], dispTolerance)):
            cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
            cv2.line(color_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
            cv2.line(color_image, centralAxis[0], centralAxis[1], (125, 125, 0), 5)
            tcp = np.array([[center_x], [center_y]], dtype=np.float32)
                
            # cv2.putText(color_image, f"Microns per pixel: {micronsOverPixels: .2f}", 
            #             (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
            
            return (tcp, edges)

    return (None, edges)