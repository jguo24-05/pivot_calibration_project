Calibrating the position of tool center point with respect to a robot base using a stereo vision system

Using the repo:

To calibrate the Basler cameras (they are calibrated already):
1. Run intrinsic_calibration/charuco_calibration.py to find each camera's internal parameters. At the top, input the settings for the charuco board. A charuco board can also be generated with intrinsic_calibration/charuco_generation.py.

2. Run extrinsic_calibration/stereo_calibration.py to find the external parameters of the 2 camera system. This will require a checkerboard of known dimensions. While calibrating, press 's' to snap images. Both cameras should have an image of the current checkerboard before moving on. To keep things consistent, also make sure that the number of images taken for both cameras is equal before moving the checkerboard to the next configuration.


To detect the tool center point:
1. Set file path settings in driver_settings.py, then run the file.
2. Run driver_to_get_images.py to grab raw images of the TCP from both cameras.
3. Run driver_to_get_segment_images.py on a GPU computer with SAM2 installed to segment the raw images.
4. Run driver_to_process_masks.py to detect the tool center point on the masked images, then triangulate them.


Important Notes: 
- Make sure the 745 camera is plugged into the first USB slot, and 746 the second. The 746 camera is calibrated as the left camera, and the 745 camera the right.
- When collecting raw TCP images, try to minimize motion blur, as the masks are difficult to process on blurred images.


Issues and Next Steps:
- Currently, the code that initially detects the tool center point has a low detection rate when the ball is not in the center of both camera frames, and it is most robust when the tool is equidistant from both cameras.

- The current code only triangulates the tool center point. It can be modified to return the central axis as well, which can be triangulated to find the orientation of the tool shaft.
- Using the detected tool center points, formulate a least-squares equation to find the transform from the end-effector to the tool center point.
- Cross-check the accuracy of the 3D triangulation with some known world points.
