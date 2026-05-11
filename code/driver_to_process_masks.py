from process_sam2_masks import *
from triangulation import *
import driver_settings


findAndWriteTCPS(driver_settings.leftMaskedDirectory, 
                 driver_settings.leftTCPJSONPath, 
                 showTCPs=driver_settings.showDetectedTCPs, 
                 isTwoMMTip=driver_settings.isTwoMMTip)

findAndWriteTCPS(driver_settings.rightMaskedDirectory, 
                 driver_settings.rightTCPJSONPath, 
                 showTCPs=driver_settings.showDetectedTCPs, 
                 isTwoMMTip=driver_settings.isTwoMMTip)
   
world_points = triangulateTCPs(driver_settings.leftTCPJSONPath, 
                               driver_settings.rightTCPJSONPath, 
                               driver_settings.targetFrames)

plotTCPs(world_points)