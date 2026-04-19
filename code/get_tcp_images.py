import pypylon.pylon as py
import numpy as np
import cv2
import json

######################### FOR OPENING STREAMS #########################
def openCamerasAndCalibrationFiles(leftExposure = 300000, rightExposure = 300000):
    NUM_CAMERAS = 2
    tlf = py.TlFactory.GetInstance()
    
    devs = tlf.EnumerateDevices()
    cam_array = py.InstantCameraArray(NUM_CAMERAS)
    
    for idx, cam in enumerate(cam_array): # type: ignore
        cam.Attach(tlf.CreateDevice(devs[idx]))
    cam_array.Open()

    # Set the exposure time for each camera and store a unique 
    # number for each camera to identify the incoming images
    cam1 = cam_array[0]
    cam1.ExposureTime.SetValue(rightExposure)
    cam1.SetCameraContext(0)
    cam2 = cam_array[1]
    cam2.ExposureTime.SetValue(leftExposure)
    cam2.SetCameraContext(1)

    # Camera Calibration
    with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix745 = np.array(json_data['mtx'])
    distCoeffs745 = np.array(json_data['dist'])

    with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix746 = np.array(json_data['mtx'])
    distCoeffs746 = np.array(json_data['dist'])

    # with open(f"./calibration_data/external_parameters.json", 'r') as file:
    #     json_data = json.load(file)
    #     F = np.array(json_data["F"])
    
    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    return (cam_array, frame_counts, converter, cameraMatrix746, distCoeffs746, cameraMatrix745, distCoeffs745)


### Given two camera feeds, snaps a photo from each. Precondition: robot is stationary ###
def snap_tcp_images(cam_array, frame_counts, converter,
                    cameraMatrix746, distCoeffs746, 
                    cameraMatrix745, distCoeffs745):
    currentCam_id = 0
    image745 = np.zeros((1670, 866, 3), dtype = np.uint8)
    image746 = np.zeros((1670, 866, 3), dtype = np.uint8)
    imageLeftFound = False
    imageRightFound = False

    while (not imageLeftFound or not imageRightFound):
        with cam_array.RetrieveResult(5000) as res:
            if res.GrabSucceeded():
                img_nr = res.ImageNumber
                cam_id = res.GetCameraContext()
                frame_counts[cam_id] = img_nr

                # Access the image data
                image = converter.Convert(res)
                color_image = image.GetArray()

                if (cam_id == 0):
                    _, color_image = undistort_image(color_image, cameraMatrix745, distCoeffs745)
                else:
                    _, color_image = undistort_image(color_image, cameraMatrix746, distCoeffs746)

                if (currentCam_id == cam_id):
                    if (currentCam_id == 0):
                        image745 = color_image
                        currentCam_id = 1 - currentCam_id
                        imageRightFound = True
                    else:
                        image746 = color_image
                        currentCam_id = 1 - currentCam_id
                        imageLeftFound = True
                   
    return (image745, image746)


### Undistorts a given image with the given distortion parameters and intrinsic matrix ###
def undistort_image(img, mtx, dst):
    h,  w = img.shape[:2]
    newcam_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    undist = cv2.undistort(img, mtx, dst, None, newcam_mtx)

    # crop the image
    x, y, w, h = roi
    undist = undist[y:y+h, x:x+w]
    return newcam_mtx, undist

    