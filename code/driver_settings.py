
# Input desired file paths here
def init():
    # raw directories: the folders where the raw images will be stored
    global leftRawDirectory 
    global rightRawDirectory
    global pointsDirectory

    leftRawDirectory = "./tcp_images/left_test"
    rightRawDirectory = "./tcp_images/right_test"
    pointsDirectory = "./detected_points_test.json"
    

    # settings for detecting the tool center point
    global isTwoMMTip 
    global showDetectedTCPs 
    global targetFrames 
    global left_initial_pts 
    global right_initial_pts 

    isTwoMMTip = True
    showDetectedTCPs = True
    targetFrames = 600
    left_initial_pts = []
    right_initial_pts = []


    # masked directories: the folders where the masked images will be stored
    global leftMaskedDirectory 
    global rightMaskedDirectory 

    leftMaskedDirectory = "./sam2_images/left_test"
    rightMaskedDirectory = "./sam2_images/right_test"


    # json paths: the json files that hold the lists of detected TCPs
    global leftTCPJSONPath
    global rightTCPJSONPath 

    leftTCPJSONPath = "./tcp_left_test.json"
    rightTCPJSONPath = "./tcp_right_test.json"


init()

    