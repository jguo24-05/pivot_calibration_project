from process_sam2_masks import *
from sam2_code import *
import driver_settings


with open(driver_settings.pointsDirectory, 'r') as json_file:
    data = json.load(json_file)
    left_initial_pts = data["left_points"]
    right_initial_pts = data["right_points"]

segment_images(video_dir = driver_settings.leftRawDirectory,
               output_dir = driver_settings.leftMaskedDirectory,
               firstPoints = left_initial_pts)

segment_images(video_dir = driver_settings.rightRawDirectory,
               output_dir = driver_settings.rightMaskedDirectory,
               firstPoints = left_initial_pts)
