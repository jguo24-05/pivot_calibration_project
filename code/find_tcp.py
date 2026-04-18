from geometric_helpers import *
from get_tcp_images import *
import numpy as np
import cv2

######################### DETECTION CODE #########################
# Calculate world coordinates based on projection matrices 
# points must be formatted as np.array([[center_x], [center_y]], dtype=np.float32)
def calculateWorldPoint(json_path, projPoints1, projPoints2):
     with open(json_path, 'r') as file:
        json_data = json.load(file)
        
        print(type(projPoints1))
        
        projMatr1 = np.array(json_data["745_projection_mtx"])
        projMatr2 = np.array(json_data["746_projection_mtx"])
        homogeneous = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)
        return (homogeneous[0], homogeneous[1], homogeneous[2]) / (homogeneous[3])


################# Attempts to find the TCP in the given image #################
def detectTCP(color_image, cannyThreshold, 
                line_thresh, circ_thresh, 
                minLineLength, maxLineGap, 
                minDistBtwnEdges, maxDistBtwnEdges, 
                minRadius, maxRadius, 
                dispTolerance, parallelTolerance):

    # 1. Preprocessing (Grayscale + Gaussian Blur)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Blur is crucial to reduce noise/specular highlights on metal surfaces
    blurred = cv2.GaussianBlur(gray, (9, 9), 5)

    # 2. Detect Drill Tip Handle Edges with Hough Probabilistic Transform 
    cannyMinThreshold = cannyThreshold/3
    edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold);
    
    lineTuple = detectLines(edges,
                            color_image, 
                            line_thresh, 
                            minLineLength, 
                            maxLineGap, 
                            minDistBtwnEdges, 
                            maxDistBtwnEdges, 
                            parallelTolerance)   
    
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
        if not (circle is None):
            center = circle[0]
            radius = circle[1]
            center_x = center[0]
            center_y = center[1]

            tcp = np.array([[center_x], [center_y]], dtype=np.float32)
                
            ## Draw the TCP in the frame ##
            micronsOverPixels = 1000/radius
            # TODO: figure out how to obtain slope properly ... a first step is flipping the image
            
            cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(color_image, (center_x, center_y), 2, (0, 0, 255), 3)
            cv2.putText(color_image, f"Microns per pixel: {micronsOverPixels: .2f}", 
                        (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
            
            return (tcp, color_image)
    
    return (None, color_image)