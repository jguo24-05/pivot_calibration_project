Calibrating the position of tool center point with respect to a robot base using a stereo vision system

1. Run the charuco calibration to find each camera's internal parameters
2. Run the stereo calibration to find the external parameters of the 2 camera system
3. Run the pivot_calibration_driver to solve for the world coordinates of each detected point relative to the camera

Important Note: 
Make sure the 745 camera is plugged into the first USB slot, and 746 the second. The 746 camera is calibrated as the left camera, and the 745 camera the right.

TODO:
Figure out slope calculations
Profile instabilities, then maybe incorporate Cotracker/Cutie/SAM2 on the camera streams