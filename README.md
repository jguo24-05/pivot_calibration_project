Calibrating the position of a robot's tool center point with a stereo vision system

1. Run the charuco calibration to find each camera's internal parameters
2. Run the stereo calibration to find the external parameters of the 2 camera system
3. Run find_tcp to solve for the world coordinates of each detected point relative to the camera

Make sure the 745 camera is plugged into the first USB slot, and 746 the second!
Very Important: the 746 camera is the left camera, and the 745 camera the right.

# TODO: 
# 0. finish copying the cotracker code
# 1. attempt to calibrate cameras again 
# 2. get cotracker videos on a grid of circles, test if the point can be tracked (must be finished by early Sunday morning)
# 3. under the assumption that the frames are matched, write code to triangulate the point
# 4. run the code and present the results (Sunday)

# Figure out slope calculation
# Incorporate Cutie or SAM2 using IMAGES as input

# Refactor so get_tcp_images takes in a Basler stream, then call get_tcp_images from find_tcp_from_video