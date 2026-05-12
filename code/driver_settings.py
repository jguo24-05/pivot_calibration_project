
# Input desired file paths here
def init():
    # raw directories: the folders where the raw images will be stored
    global leftRawDirectory 
    global rightRawDirectory
    global pointsDirectory

    leftRawDirectory = "./initial_images/left_test_0"
    rightRawDirectory = "./initial_images/right_test_0"
    pointsDirectory = "./detected_points/initial_points_test_0.json"
    

    # settings for detecting the tool center point
    global isTwoMMTip 
    global showDetectedTCPs 
    global targetFrames 
    global left_initial_pts 
    global right_initial_pts 

    isTwoMMTip = True
    showDetectedTCPs = True
    targetFrames = 300
    left_initial_pts = []
    right_initial_pts = []


    # masked directories: the folders where the masked images will be stored
    global leftMaskedDirectory 
    global rightMaskedDirectory 

    leftMaskedDirectory = "./sam2_images/left_test_0"
    rightMaskedDirectory = "./sam2_images/right_test_0"


    # json paths: the json files that hold the lists of detected TCPs
    global leftTCPJSONPath
    global rightTCPJSONPath 

    leftTCPJSONPath = "./detected_points/tcp_left_test_0.json"
    rightTCPJSONPath = "./detected_points/tcp_right_test_0.json"


init()

    