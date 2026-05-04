from process_sam2_masks import *
from get_tcp_videos import *
from triangulation import *
# from sam2_code import *

leftRawDirectory = "./code/left_test_1"
rightRawDirectory = "./code/right_test_1"
pointsDirectory = "./detected_points_test1.json"

leftMaskedDirectory = "./sam2_images/left_test_1"
rightMaskedDirectory = "./sam2_images/right_test_1"

leftTCPJSONPath = "./tcp_left_test1.json"
rightTCPJSONPath = "./tcp_right_test1.json"

isTwoMMTip = True
showDetectedTCPs = True
targetFrames = 600

left_initial_pts = []
right_initial_pts = []

# getRawTCPImages(leftRawDirectory,
#                 rightRawDirectory,
#                 pointsDirectory,
#                 targetFrames=targetFrames,
#                 exposureLeft=200000,
#                 exposureRight=200000,
#                 is2MMTip=isTwoMMTip)

# with open(pointsDirectory, 'r') as json_file:
#     data = json.load(json_file)
#     left_initial_pts = data["left_points"]
#     right_initial_pts = data["right_points"]

# segment_images(video_dir = leftRawDirectory,
#                output_dir = leftMaskedDirectory,
#                firstPoints = left_initial_pts)
# segment_images(video_dir = rightRawDirectory,
#                output_dir = rightMaskedDirectory,
#                firstPoints = left_initial_pts)

# findAndWriteTCPS(leftMaskedDirectory, leftTCPJSONPath, showTCPs=showDetectedTCPs, isTwoMMTip=isTwoMMTip)
# findAndWriteTCPS(rightMaskedDirectory, rightTCPJSONPath, showTCPs=showDetectedTCPs, isTwoMMTip=isTwoMMTip)
   

world_points = triangulateTCPs(leftTCPJSONPath, rightTCPJSONPath, targetFrames)
plotTCPs(world_points)

# directory = "./sam2_images/left_test_1/00068.png"
# color_image = cv2.imread(directory)

# ### Set detection parameters ###
# # Canny Threshold
# cannyThreshMin = 25
# cannyThreshMax = 100
# # Line Accumulator Matrix
# lineAccMin = 100
# lineAccMax = 300
# # Circle Accumulator Matrix
# circleAccMin = 15
# circleAccMax = 300
# # Minimum circle radius
# minRadiusMin = 30
# minRadiusMax = 80
# # Maximum circle radius
# maxRadiusMin = 100
# maxRadiusMax = 200
# # Minimum line length
# minLineMin = 10
# minLineMax = 100
# # Maximum line gap
# maxLineMin = 50
# maxLineMax = 250
# # Minimum distance between edges
# minDistMin = 20
# minDistMax = 100
# # Maximum distance between edges      
# maxDistMin = 2000
# maxDistMax = 2500
# # Error for tip and axis alignment
# errorMin = 20
# errorMax = 100


# ## Window 1 - Left Camera Feed ##
# cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Window 1", 700, 700)
# win1_name = "Window 1"


# cv2.createTrackbar('Canny Threshold', win1_name, cannyThreshMin, cannyThreshMax, lambda a: None);
# # Accumulator Threshold for Lines
# cv2.createTrackbar('Line Accumulator Threshold', win1_name, lineAccMin, lineAccMax, lambda a: None);
# # Circle Radius Accumulator Threshold
# cv2.createTrackbar('Circle Accumulator Threshold', win1_name, circleAccMin, circleAccMax, lambda a: None)
# # Min Radius: Minimum circle radius to detect
# cv2.createTrackbar('Minimum Radius', win1_name, minRadiusMin, minRadiusMax, lambda a: None)
# # Max Radius: Maximum circle radius to detect
# cv2.createTrackbar('Maximum Radius', win1_name, maxRadiusMin, maxRadiusMax, lambda a: None)
# # Minimum Line Length: Minimum line length to detect
# cv2.createTrackbar('Minimum Line Length', win1_name, minLineMin, minLineMax, lambda a: None)
# # Maximum Line Gap: Maximum allowed gap in a line
# cv2.createTrackbar('Maximum Line Gap', win1_name, maxLineMin, maxLineMax, lambda a: None)
# # Minimum Distance Between Edge Lines
# cv2.createTrackbar('Minimum Distance Between Edges', win1_name, minDistMin, minDistMax, lambda a: None)
# # # Minimum Distance Between Edge Lines
# cv2.createTrackbar('Maximum Distance Between Edges', win1_name, maxDistMin, maxDistMax, lambda a: None)
# # Tolerance for how far the radius can be from the detected central axis
# cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win1_name, errorMin, errorMax, lambda a: None)


# while (True):
#     # Get left camera trackbar positions
#     cannyThreshold = cv2.getTrackbarPos('Canny Threshold', "Window 1")    
#     line_thresh = cv2.getTrackbarPos('Line Accumulator Threshold', "Window 1")
#     circ_thresh = cv2.getTrackbarPos('Circle Accumulator Threshold', "Window 1")
#     minLineLength = cv2.getTrackbarPos('Minimum Line Length', "Window 1")
#     maxLineGap = cv2.getTrackbarPos('Maximum Line Gap', "Window 1")
#     minDistBtwnEdges = cv2.getTrackbarPos('Minimum Distance Between Edges', "Window 1")
#     maxDistBtwnEdges = cv2.getTrackbarPos('Maximum Distance Between Edges', "Window 1")
#     minRadius = cv2.getTrackbarPos('Minimum Radius', "Window 1")
#     maxRadius = cv2.getTrackbarPos('Maximum Radius', "Window 1")
#     dispTolerance = cv2.getTrackbarPos('Maximum Error for Tip and Axis Alignment', "Window 1")


#     # (tcp, centralAxisMidpoint, img, edges) = detectTCP(color_image, cannyThreshold,
#     #     line_thresh, circ_thresh,
#     #     minLineLength, maxLineGap,
#     #     minDistBtwnEdges, maxDistBtwnEdges,
#     #     minRadius, maxRadius,
#     #     dispTolerance, parallelTolerance=.85)
    
#     (center, drawing, img) = findTCPFromMask(directory, isTwoMMTip)
   
#     if (img is not None and drawing is not None):
#         cv2.imshow(win1_name, drawing)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#         break

