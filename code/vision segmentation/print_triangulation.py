from stereo_calibration import *
## with the points, run triangulation and print to video ##


## Assuming points745 and points746 are of the form
## [[x1, y1], ..., [xn, yn]]
def triangulateAndDraw(video1Path, outputVideoPath, points745, points746):
    cap = cv2.VideoCapture(video1Path)
    output = cv2.VideoWriter(outputVideoPath,
                             cv2.VideoWriter.fourcc(*'MPEG'),
                             30,
                             (1080,1920)) # TODO: change the fps and frame rate accordingly
    worldPoints = []

    frames = min(points745.length, points746.length)
    for f in range(frames):
        point746 = np.array([[points746[f][0], points746[f][1]]], dtype=np.float32)
        point745 = np.array([[points745[f][0], points745[f][1]]], dtype=np.float32)
        triangulated = grab3DPoints("./calibration_data/external_parameters.json", point746, point745)
        worldPoints.append([triangulated[0][0], triangulated[1][0], triangulated[2][0]])

    frameCounter = 0
    while (True):
        ret, frame = cap.read()
        if (ret):
            cv2.putText(frame, f"Position (mm): ({worldPoints[frameCounter][0]: .1f}, {worldPoints[frameCounter][1]: .1f}, {worldPoints[frameCounter][2]: .1f})",
                                            (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            # writing the new frame in output
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    cv2.destroyAllWindows()
    output.release()
    cap.release()




