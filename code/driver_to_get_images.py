from get_tcp_videos import *
import driver_settings


getRawTCPImages(driver_settings.leftRawDirectory,
                driver_settings.rightRawDirectory,
                driver_settings.pointsDirectory,
                targetFrames=driver_settings.targetFrames,
                exposureLeft=200000,
                exposureRight=200000,
                is2MMTip=driver_settings.isTwoMMTip)
