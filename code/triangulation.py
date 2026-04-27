import json
from find_tcp import *

##### Triangulating the detected TCPs #####

left_json_path = "./tcp_left_example1"
right_json_path = "./tcp_right_example1"

left_points = {}
right_points = {}
with open(left_json_path) as json_file:
    left_points = json.load(json_file)

with open(right_json_path) as json_file:
    right_points = json.load(json_file)

frames = 300
world_points = []
for i in range(frames):
    key = f'{i}'
    if (key in left_points.keys() and key in right_points.keys()):
        # np.array([[center_x], [center_y]], dtype=np.float32)
        left = left_points[key]
        right = right_points[key]

        left_pt = np.array([[left[0]], [left[1]]], dtype = np.float32)
        right_pt = np.array([[right[0]], [right[1]]], dtype = np.float32)
        worldPoint = calculateWorldPoint('./calibration_data/external_parameters.json', left_pt, right_pt)
        world_points.append(worldPoint)

# TODO: plot world_points with matplotlib 3d