import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from cotracker.predictor import CoTrackerPredictor
from print_triangulation import *


# video = read_video_from_path('./assets/apple.mp4')

def show_video(video_path):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")
 

def track_point_in_video(videoPath, finalVideoName, listOfPoints):
    video = read_video_from_path(videoPath)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    listOfPoints = listOfPoints.astype(np.float32)

    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/scaled_offline.pth'
        )
    )

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()

    # queries = torch.tensor([
    #     [0., 400., 350.],  # point tracked from the first frame
    #     [10., 600., 500.], # frame number 10
    #     [20., 750., 600.], # ...
    #     [30., 900., 200.]
    # ])

    queries = torch.tensor(listOfPoints)

    if torch.cuda.is_available():
        queries = queries.cuda()

    pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)

    vis = Visualizer(
        save_dir='./videos',
        linewidth=6,
        mode='cool',
        tracks_leave_trace=-1
    )

    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=finalVideoName)

    show_video(f"{finalVideoName}.mp4")
    return pred_tracks


#### Driver Code ####
cam_745_points = []
cam_746_points = []

with open("C:/Users/ninig/JHU_Work/spring26/lcsr/calibration-code/detected_points.json", 'r') as file:
        json_data = json.load(file)
        cam_745_points = np.array(json_data["cam_745"])
        cam_746_points = np.array(json_data["cam_746"])

pred_tracks_745 = track_point_in_video("C:/Users/ninig/JHU_Work/spring26/lcsr/calibration-code/camera_745_output.mp4", "C:/Users/ninig/JHU_Work/spring26/lcsr/calibration-code/tracked_745_output", cam_745_points)
# pred_tracks_746 = track_point_in_video("C:/Users/ninig/JHU_Work/spring26/lcsr/calibration-code/camera_746_output.mp4", "tracked_746_output", cam_745_points)

# triangulateAndDraw("tracked_745_output", "annotated_745_output", pred_tracks_745, pred_tracks_746)
### TODO: figure out the format of pred_tracks_745

print(pred_tracks_745)

