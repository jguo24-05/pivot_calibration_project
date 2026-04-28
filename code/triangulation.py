import json
from find_tcp import *
import matplotlib.pyplot as plt
import matplotlib as mpl

##### Triangulating the detected TCPs #####
def triangulateTCPs(left_json_path, right_json_path, numFrames):
    left_points = {}
    right_points = {}
    with open(left_json_path) as json_file:
        left_points = json.load(json_file)

    with open(right_json_path) as json_file:
        right_points = json.load(json_file)

    world_points = []
    for i in range(numFrames):
        key = f'{i}'
        if (key in left_points.keys() and key in right_points.keys()):
            left = left_points[key]
            right = right_points[key]

            left_pt = np.array([[left[0]], [left[1]]], dtype = np.float32)
            right_pt = np.array([[right[0]], [right[1]]], dtype = np.float32)
            worldPoint = calculateWorldPoint('./calibration_data/external_parameters.json', left_pt, right_pt)
            world_points.append(worldPoint)

    return world_points

### Plotting the computed TCPs ###
def plotTCPs(world_points):
    ### Plotting the 3D points with Matplotlib ###
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = mpl.colormaps['viridis']

    for i in range(len(world_points)):
        color = cmap(i / len(world_points))
        world_pt = world_points[i]
        x = world_pt[0]
        y = world_pt[1]
        z = world_pt[2]
        ax.scatter(x, y, z, color = color)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()