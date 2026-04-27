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
    if (are_lines_collinear(line1[0], line2[0], 15)):
        return 0
    else:
        epsilon = 10^-9
        ax1, ay1, ax2, ay2 = line1[0]
        bx1, by1, bx2, by2 = line2[0]

        if (ax2-ax1 < epsilon and bx2-bx1 < epsilon):
            return abs(ay2-by2)
        elif (ay2-ay1 < epsilon and by2-by1 < epsilon):
            return abs(ax2-bx2)
        else:
            m1 = (ay2-ay1) / float((ax2-ax1))
            m2 = (by2-by1) / float((bx2-bx1))
            c1 = ay1 - m1*ax1
            c2 = by1 - m2*bx1
            return pow(pow((m1-m2), 2) + pow(c1-c2, 2), 0.5)
    

def point_to_line_dist(point, line_start, line_end):
    p = np.array(point)
    s = np.array(line_start)
    e = np.array(line_end)
    
    # Vector of the line segment
    line_vec = e - s
    # Vector from start to point
    point_vec = p - s
    
    line_len_sq = np.sum(line_vec**2)
    
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)
    
    # Projection factor (clamped between 0 and 1)
    t = np.dot(point_vec, line_vec) / line_len_sq
    
    return np.linalg.norm(point_vec - t * line_vec)


def are_lines_collinear(line1, line2, dist_threshold):
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, _, _ = line2
    return point_to_line_dist((bx1,by1), (ax1, ay1), (ax2, ay2)) < dist_threshold


# Detect lines using Probabilistic Hough Transform
def detectLines(edges, color_image, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, parallelTolerance):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap) 

    if lines is not None:
        if (len(lines) < 2):
            return None    # skip this frame if fewer than two lines were found 
    
        for i in range(len(lines)-1):  
            # Debugging 
            # ax1, ay1, ax2, ay2 = lines[i][0]
            # cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
                
            for j in range(i, len(lines)):
                # Debugging
                # bx1, by1, bx2, by2 = lines[j][0]
                # cv2.line(color_image, (bx1, by1), (bx2, by2), (255, 0, 0), 5)   

                line1 = lines[i]
                line2 = lines[j]
                
                rSlope = slopeRatio(line1, line2)
                distBtwnEdges = distBetweenLines(line1, line2)
                
                if ((rSlope < 1 + parallelTolerance) and (rSlope > 1 - parallelTolerance)):
                    if (distBtwnEdges > minDistBtwnEdges and distBtwnEdges < maxDistBtwnEdges): 
                        return (line1, line2)
                #     else:
                #         print(f"Distance issue: {distBtwnEdges}")
                # else:
                #     print("Slope issue")
        
        return None     # no parallel lines were found this frame


# Detect lines using Probabilistic Hough Transform
def detectLinesAlt(edges, color_image, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, parallelTolerance):
    result = []
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap) 

    if lines is not None:
        if (len(lines) < 2):
            return result    # skip this frame if fewer than two lines were found 
    
        for i in range(len(lines)-1):  
            # Debugging 
            # ax1, ay1, ax2, ay2 = lines[i][0]
            # cv2.line(edges, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
                
            for j in range(i, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                ax1, ay1, _, _ = line1[0]
                width = color_image.shape[1]
                height = color_image.shape[0]
                if (ax1 < width*0.25 or ax1 > width*0.75 or ay1 < height * 0.3):  # TODO: Rule out lines too close to the edge
                    continue

                rSlope = slopeRatio(line1, line2)
                distBtwnEdges = distBetweenLines(line1, line2)
                
                if ((rSlope < 1 + parallelTolerance) and (rSlope > 1 - parallelTolerance)):
                    ax1, ay1, ax2, ay2 = line1[0]
                    # cv2.line(edges, (ax1, ay1), (ax2, ay2), (255, 0, 0), 7)    
                    bx1, by1, bx2, by2 = line2[0]
                    # cv2.line(edges, (bx1, by1), (bx2, by2), (255, 0, 0), 1) 

                    if (distBtwnEdges > minDistBtwnEdges and distBtwnEdges < maxDistBtwnEdges): 
                        result.append((line1, line2))
        
    return result     # no parallel lines were found this frame


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

