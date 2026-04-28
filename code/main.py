from process_sam2_masks import *
from get_tcp_videos import *
from triangulation import *
# from sam2_code import *

leftRawDirectory = "./code/left_example_2"
rightRawDirectory = "./code/right_example_2"
pointsDirectory = "./detected_points_example2.json"

leftMaskedDirectory = "./sam2_images/left_example_2"
rightMaskedDirectory = "./sam2_images/right_example_2"

leftTCPJSONPath = "./tcp_left_example2.json"
rightTCPJSONPath = "./tcp_right_example2.json"

isTwoMMTip = True
showDetectedTCPs = True
targetFrames = 300

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

findAndWriteTCPS(leftMaskedDirectory, leftTCPJSONPath, showTCPs=showDetectedTCPs, isTwoMMTip=isTwoMMTip)
findAndWriteTCPS(rightMaskedDirectory, rightTCPJSONPath, showTCPs=showDetectedTCPs, isTwoMMTip=isTwoMMTip)

world_points = triangulateTCPs(leftTCPJSONPath, rightTCPJSONPath, targetFrames)
plotTCPs(world_points)