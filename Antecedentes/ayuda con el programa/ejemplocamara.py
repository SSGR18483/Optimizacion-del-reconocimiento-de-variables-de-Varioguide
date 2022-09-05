# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:41:39 2021

@author: joe
"""

from goprocam import GoProCamera, constants
gopro = GoProCamera.GoPro(constants.gpcontrol)
gopro.overview()

def take_photo():
    gopro.take_photo(timer=1)
    gopro.downloadLastMedia(custom_filename="selfie.png")
    gopro.delete("last")

take_photo()
gopro.listMedia(True)
    
