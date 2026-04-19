import cv2
import numpy as np

######################### GEOMETRIC HELPERS #########################
# absolute ratio of slopes
def slopeRatio(line1, line2):
    epsilon = 10**-15
    x1, y1, x2, y2 = line1[0]
    u1, v1, u2, v2 = line2[0]
    if (abs(x2 - x1) < epsilon and abs(u2 - u1) < epsilon): return 10**9    # both vertical - rule out
    elif (abs(y2 - y1) < epsilon and abs(v2 - v1) < epsilon): return 10**9  # both horizontal - rule out
    elif (abs(x2 - x1) < epsilon or abs(u2 - u1) < epsilon): return 10**9
    elif (abs(y2 - y1) < epsilon or abs(v2 - v1) < epsilon): return 10**9
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (v2 - v1) / (u2 - u1)
    return abs(slope1 / slope2)
    
   
# Checks if point is on the line defined by (endpoint1, endpoint2) with the given tolerance
def pointOnLine(point, endpoint1, endpoint2, tolerance):
    if ((endpoint2[0] - endpoint1[0]) == 0):    # vertical line
        return point[0] == endpoint2[0]
    
    m = (endpoint2[1] - endpoint1[1]) / (endpoint2[0] - endpoint1[0])
    c = endpoint1[1] - m * endpoint1[0]
    expectedY = point[0] * m + c
    return abs(expectedY - point[1]) < tolerance


# Returns the euclidean distance between <m1, c1> and <m2, c2>
def distBetweenLines(line1, line2): 
    epsilon = 10**-6
    ax1, ax2, ay1, ay2 = line1[0]
    bx1, bx2, by1, by2 = line2[0]
    if (abs(ax2-ax1) < epsilon and abs(bx2-bx1) < epsilon):    # vertical lines
        return abs(bx1-ax1)
    elif (abs(ax2-ax1) < epsilon or abs(bx2-bx1) < epsilon):   # only one vertical line
        return 10**9    # just a huge number so tests fail
    else:
        m1 = (ay2-ay1) / float((ax2-ax1))
        m2 = (by2-by1) / float((bx2-bx1))
        c1 = ay1 - m1*ax1
        c2 = by1 - m2*bx1
        return pow(pow((m1-m2), 2) + pow(c1-c2, 2), 0.5)


# Detect lines using Probabilistic Hough Transform
def detectLines(edges, color_image, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, parallelTolerance):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap) 

    if lines is not None:
        if (len(lines) < 2):
            return None    # skip this frame if fewer than two lines were found 
    
        for i in range(len(lines)-1):  
            # Debugging 
            # ax1, ay1, ax2, ay2 = lines[i][0]
            # cv2.line(edges, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
                
            for j in range(i, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                ax1, ay1, ax2, ay2 = line1[0]
                width = color_image.shape[1]
                height = color_image.shape[0]
                if (ax1 < width*0.1 or ax1 > width*0.9 or ay1 > height*0.9):  # TODO: Rule out lines too close to the edge
                    continue

                rSlope = slopeRatio(line1, line2)
                distBtwnEdges = distBetweenLines(line1, line2)
                
                if ((rSlope < 1 + parallelTolerance) and (rSlope > 1 - parallelTolerance)):
                    ax1, ay1, ax2, ay2 = line1[0]
                    # cv2.line(edges, (ax1, ay1), (ax2, ay2), (255, 0, 0), 7)    
                    bx1, by1, bx2, by2 = line2[0]
                    # cv2.line(edges, (bx1, by1), (bx2, by2), (255, 0, 0), 1) 

                    if (distBtwnEdges > minDistBtwnEdges and distBtwnEdges < maxDistBtwnEdges): 
                        return (line1, line2)
        
        return None     # no parallel lines were found this frame


# Detect circles that lie on the given centralAxis
def detectCircles(grayFrame, accumulatorRes, minDist, cannyThreshold, circAccThreshold, minRadius, maxRadius, centralAxis, dispTolerance):
    # Detect circles with the Hough transform
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 
                            dp = accumulatorRes, minDist = minDist, param1 = cannyThreshold,   # for hough_gradient_alt 
                            param2 = circAccThreshold, minRadius = minRadius, 
                            maxRadius = maxRadius)

    if not (circles is None):
        circles = np.round(circles).astype("uint16")
        for i in circles[0,:]: 
            center = (i[0], i[1])
            radius = i[2]

            if (pointOnLine(center, centralAxis[0], centralAxis[1], dispTolerance)):
                return (center, radius)

