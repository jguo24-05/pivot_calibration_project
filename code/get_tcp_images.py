import pypylon.pylon as py
from charuco_calibration import *
import numpy as np
import cv2
import json

######################### DETECTION LOOP #########################
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

    with open(f"./calibration_data/external_parameters.json", 'r') as file:
        json_data = json.load(file)
        F = np.array(json_data["F"])
    
    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    return (cam_array, frame_counts, converter, cameraMatrix746, distCoeffs746, cameraMatrix745, distCoeffs745, F)


### Opens a camera feed and saves calibration images when 's' is pressed ###
def save_TCP_images(imagePathLeft, imagePathRight):
    (cam_array, frame_counts, converter,
     cameraMatrix746, distCoeffs746, 
     cameraMatrix745, distCoeffs745, F) = openCamerasAndCalibrationFiles()
    
    currentCam_id = 0

    cam_array.StartGrabbing()
    while True:
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

                key = cv2.waitKey(1) & 0xFF
                if (currentCam_id == cam_id and key == ord('s')):       # S Key pressed
                    if (currentCam_id == 0):
                        cv2.imwrite(f'{imagePathRight}.png', color_image)
                        currentCam_id = 1
                    else:
                        cv2.imwrite(f'{imagePathLeft}.png', color_image)
                        currentCam_id = 0
                   
                if (cam_id == 0):
                    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Window 1', 700, 500)
                    cv2.imshow('Window 1', color_image)
                    
                elif (cam_id == 1):
                    cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Window 2', 700, 500)
                    cv2.imshow('Window 2', color_image)
                
                # check if q button has been pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    cam_array.StopGrabbing()
    cam_array.Close()


### Given two camera feeds, snaps a photo from each. Precondition: robot is stationary ###
def snap_tcp_images(cam_array, frame_counts, converter,
                    cameraMatrix746, distCoeffs746, 
                    cameraMatrix745, distCoeffs745):
    currentCam_id = 0
    image745 = []
    image746 = []

    cam_array.StartGrabbing()
    while (len(image745) == 0 or len(image746) == 0):
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
                    else:
                        image746 = color_image
                        currentCam_id = 1 - currentCam_id
                   
    cam_array.StopGrabbing()
    cam_array.Close()
    return (image745, image746)
    