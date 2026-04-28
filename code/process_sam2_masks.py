from find_tcp import *
from geometric_helpers import *
import glob


### Detect TCP Using Bounding Boxes ###
def findTCPFromMask(filename, isTwoMMTip):
    ### Detection Parameters ###
    # Canny Threshold
    cannyThresh = 30
    # Circle Accumulator Matrix
    circleAcc = 15 
    # Minimum circle radius
    minRadius = 50   # Note: should be 30 for 2mm tip, 50 for 4mm tip
    if (isTwoMMTip):
        minRadius = 30
    # Maximum circle radius
    maxRadius = 100

    color_image = cv2.imread(filename)
    if color_image is not None:
        img = color_image.copy()

        # 1. Preprocessing (Grayscale)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 2. Finding contours of tool
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tool_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(tool_contour, True)
        smooth_edges = cv2.approxPolyDP(tool_contour, epsilon, True)
        
        # 3. Drawing the contours onto a black background
        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(drawing, [smooth_edges], -1, (0, 0, 255), 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        drawing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)    # for smoothing the ridges of the larger drill
        grayContours = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        
        # 4. Finding the edges of the tool
        toolEdges = detectLines(grayContours, drawing, 
                                line_thresh=50, 
                                minLineLength=100, maxLineGap=10, 
                                minDistBtwnEdges=2, maxDistBtwnEdges=2000,
                                parallelTolerance=0.5)
        
        if (toolEdges is None):
            cv2.putText(img, "The tool shaft was not detected this frame", (70, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
        if (toolEdges is not None):
            line1 = toolEdges[0]
            line2 = toolEdges[1]
            ax1, ay1, ax2, ay2 = line1[0]
            cx1, cy1, cx2, cy2 = line2[0]
            
            bx1 = int((ax1+cx1)/2.0)
            by1 = int((ay1+cy1)/2.0)
            bx2 = int((ax2+cx2)/2.0)
            by2 = int((ay2+cy2)/2.0)
            centralAxis = ((bx1, by1), (bx2, by2))

            cv2.line(drawing, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
            cv2.line(drawing, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
            cv2.line(drawing, centralAxis[0], centralAxis[1], (125, 125, 0), 5)
            cv2.line(img, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
            cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
            cv2.line(img, centralAxis[0], centralAxis[1], (125, 125, 0), 5)
            
            # 5. Detecting the circle of largest radius on the contours
            accumulatorRes = 1     
            minDist = 2000
            circles = cv2.HoughCircles(grayContours, cv2.HOUGH_GRADIENT, 
                                        dp = accumulatorRes, minDist = minDist, param1 = cannyThresh, 
                                        param2 = circleAcc, minRadius = minRadius, 
                                        maxRadius = maxRadius)
            
            if not (circles is None):
                circles = np.round(circles).astype("uint16")
                for i in circles[0,:]: 
                    center = (i[0], i[1])
                    center_x = center[0]
                    center_y = center[1]
                    radius = i[2]
                    cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
                    cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), 3)

                    if (pointOnLine(center, centralAxis[0], centralAxis[1], tolerance=20)):
                        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
                        cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), 3)
                        cv2.circle(drawing, (center_x, center_y), radius, (0, 255, 0), 2)
                        cv2.circle(drawing, (center_x, center_y), 2, (255, 255, 0), 3)
                        # print(f"Radius: {radius}")
                        return (center, drawing, img)
            else:
                print("No circles :(")
        return (None, drawing, img)
    return (None, color_image, color_image)


### Test on single image ###
def testSingleImage(filename, isTwoMMTip):
    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 700, 700)
    win1_name = "Window 1"

    (center, drawing, img) = findTCPFromMask(filename, isTwoMMTip)
    if (drawing is not None):
            cv2.imshow(win1_name, drawing)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()


### Finds TCPs in all images of the given directory and writes to the jsonPath ###
def findAndWriteTCPS(directory, jsonPath, showTCPs, isTwoMMTip):
    ## Process SAM2 Masks to find the TCP ##
    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 700, 700)
    win1_name = "Window 1"

    detectedFrames = 0
    points_dict = {}
    imgDirectory = f'{directory}/*.png'

    for filename in glob.glob(imgDirectory):
        (center, drawing, img) = findTCPFromMask(filename, isTwoMMTip)
        if (center is not None):
            points_dict[detectedFrames] = [int(center[0]), int(center[1])]
            detectedFrames += 1
        if (showTCPs and img is not None and drawing is not None):
            cv2.imshow(win1_name, img)
            cv2.waitKey(1)

            # For clicking through the images one by one
            # key = cv2.waitKey(0)
            # if key == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break

    # Calibration Log
    # with open("./calibration_log.txt", "a") as f:
    #     f.write(imgDirectory + "\n")
    #     f.write("-" * 20 + "\n") 
    #     f.write(f"{detectedFrames} frames detected out of 300\n")
    #     f.write("-" * 20 + "\n") 

    # Write the detected points to a json file
    with open(jsonPath, 'w') as f:
        json.dump(points_dict, f, indent=4)

    cv2.destroyAllWindows()