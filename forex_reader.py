#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:53:15 2017

@author: fosa
"""


import pandas as pd
import numpy as np
from PIL import Image


import io
from scipy import misc

from timezone import LocalTimezone
from datetime import tzinfo, timedelta, datetime
import time as _time
import os 


import random



def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue


def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    #print (red, green, blue)
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint
    

def numberToColor(number):
    number *= 10000 #integer form
    color = getRGBfromI(int(number))
    r = color[0]
    g = color[1]
    b = color[2]
    color_np = np.array([[[r,g,b]]], dtype=np.float32)
    return color_np

  
def getImageArray(datasets, look_ahead=1, first_index=100):
    full_2d_list = []
    #last_index = 100
    
    for k in range(len(datasets)):
        
        colors_list_high = []
        colors_list_low = []
        for i in range(100):
            high = datasets[k].iloc[i + first_index]["high"]
            low = datasets[k].iloc[i + first_index]["low"]

            colors_list_high.append(numberToColor(high))
            colors_list_low.append(numberToColor(low))

        colors_array_h = np.hstack(colors_list_high)
        colors_array_l = np.hstack(colors_list_low)
        
        full_2d_list.append(colors_array_h)
        full_2d_list.append(colors_array_l)
    full_2d_colors_array = np.concatenate(full_2d_list)
    full_2d_colors_array = full_2d_colors_array.transpose(2,0,1)
    
    target_high = datasets[0].iloc[100 + first_index + look_ahead]["high"]        
    adjusted_target_value = target_high / (10 ** len(str(int(target_high))))
    
    return full_2d_colors_array, adjusted_target_value


def loadFilledFiles():
    files = [f for f in os.listdir('Datasets/Filled') if f.endswith(".csv")]
    datasets = [pd.read_csv('Datasets/Filled/' + f, index_col=0) for f in files]
    return files, datasets
#
#files, datasets = loadFilledFiles()
#color_array, adjusted_targets = getImageArray(datasets)

#full_2d_colors_array = full_2d_colors_array.transpose(1,2,0)
#x = Image.fromarray(np.uint8(full_2d_colors_array))
#size = 300, 180
#x.resize(size, Image.ANTIALIAS)
#
#x.show()
