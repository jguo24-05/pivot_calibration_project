Calibrating the position of tool center point with respect to a robot base using a stereo vision system

1. Run the charuco calibration to find each camera's internal parameters
2. Run the stereo calibration to find the external parameters of the 2 camera system
3. Run the main file to grab raw images, process them with SAM2, find the TCPs, and solve for the world points

Important Note: 
Make sure the 745 camera is plugged into the first USB slot, and 746 the second. The 746 camera is calibrated as the left camera, and the 745 camera the right.