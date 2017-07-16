#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:41:19 2017

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

from forex_model import *
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
    print red, green, blue
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint
    



def insertMissingRows(x1, x2, dataset):
    start_row = dataset.iloc[x1]
    end_row = dataset.iloc[x2]

    start_time = datetime.strptime(start_row["datetimes"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_row["datetimes"], "%Y-%m-%d %H:%M:%S")
    
    time_difference = (end_time - start_time).seconds / 60
    open_increment = (end_row["open"] - start_row["open"]) / time_difference
    high_increment = (end_row["high"] - start_row["high"]) / time_difference
    low_increment = (end_row["low"] - start_row["low"]) / time_difference
    close_increment = (end_row["close"] - start_row["close"]) / time_difference
    #print("close increment", close_increment)
    #print(start_row["close"], start_row["close"] + close_increment)
    
    
    columns = [ "datetimes", "waste", "open", "high", "low", "close"]

    temp_df = pd.DataFrame(columns=columns)
    
    for i in range((time_difference) - 1):
        new_line = pd.DataFrame({
                              "datetimes": start_time + timedelta(0, (60 * int(i + 1))),
                              "waste": 0,
                              "open" : start_row["open"] + open_increment * (i + 1),
                              "high" : start_row["high"] + high_increment * (i + 1),
                              "low" : start_row["low"] + low_increment * (i + 1),
                              "close" : start_row["close"] + close_increment * (i + 1)}, index=[0])
        temp_df = pd.concat([temp_df, new_line], ignore_index=True)
        
    dataset = pd.concat([dataset.ix[:x1], temp_df, dataset.ix[x2:]]).reset_index(drop=True)
    
    return dataset, time_difference - 1 #returning T-diff so we can advance by that many

def fillAllMissingRows(dataset):
    for i in range(len(dataset) - 1):
        advance_by_rows = 0
        if i % 10000 == 0:
            print("row ", i)
        try:
            start_time = datetime.strptime(dataset.iloc[i]["datetimes"], "%Y-%m-%d %H:%M:%S")
        except TypeError:
            start_time = dataset.iloc[i]["datetimes"]
        try:
            end_time = datetime.strptime(dataset.iloc[i+1]["datetimes"], "%Y-%m-%d %H:%M:%S")
        except TypeError:
            end_time = dataset.iloc[i+1]["datetimes"]

        if (end_time - start_time).seconds != 60:
            dataset, advance_by_rows = insertMissingRows(i, i+1, dataset)
        i += advance_by_rows
    return dataset

    
def loadFiles():
    files = [f for f in os.listdir('Datasets') if f.endswith(".csv")]
    datasets = [pd.read_csv('Datasets/' + f) for f in files]
    return files, datasets
        

    
def saveDataset(dataset, name):
    print("saving ", name)
    dataset.to_csv('Datasets/Filled/' + name + "filled.csv")    

    
    

def fillDatasets():
    files, datasets = loadFiles()
    
    for i in range(len(files)):
        
        print("filling ", i , " ", files[i])
        dataset = fillAllMissingRows(datasets[i])
        saveDataset(dataset, files[i])
        

def loadFilledFiles():
    files = [f for f in os.listdir('Datasets/Filled') if f.endswith(".csv")]
    datasets = [pd.read_csv('Datasets/Filled/' + f, index_col=0) for f in files]
    return files, datasets


def chopDatasets(datasets, limit):
    for i in range(len(datasets)):
        datasets[i] = datasets[i].ix[:limit]
    return datasets
    
#dataset_100 = dataset.ix[:1000]

#dataset_filled = checkMissingRows(dataset_100)


#training data can be generated simultaneously as the target values
#next lets make a functionto load each of the datasets in filled, and then combine
#into a single array

def numberToColor(number):
    number *= 10000 #integer form
    color = getRGBfromI(int(number))
    r = color[0]
    g = color[1]
    b = color[2]
    color_np = np.array([[[r,g,b]]], dtype=np.float32)
    return color_np

def createTargetArrays():
    batch_size = 1000  
    batches = len(datasets[0]) / batch_size
    
    for i in range(batches - 1):
        targets = getTargets((100 + i * batch_size), batch_size=batch_size)
        array = np.array(targets, dtype=np.float32)
        np.save("Training/targets_array " + str(i), array)
        
    
def getTargets(start_index, look_ahead=1, batch_size=1000):
    targets = []
    constant_reduction = 10000
    for i in range(batch_size):    
        target_high = datasets[0].iloc[i + 100 + look_ahead]["high"]
        print("target_high", target_high)
        target_high /= constant_reduction
        targets.append([target_high])
    return targets
    
    
def getImageArray(look_ahead=1, first_index=100):
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

def makeModel():
    regression_output = 1 

    model = CNNSmall(regression_output)
    regression_model = regressionLoss(model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    #classifier_model.to_gpu(0)
    
    return regression_model, optimizer, model

def train(iterations, x_gpu, target_gpu):
    for i in range(iterations):
        regression_model.cleargrads()
        
        #cpu version: loss = classifier_model(image_array, target_values)
        loss = regression_model(x_gpu, target_gpu)
        loss.backward()
        #print('loss 1' , loss[0].data)
        
        optimizer.update()
        #if i % 1000 == 0:
        print("iteration", i,
              'loss' , loss.data)
              
def batcher():
    batches = len(datasets[0]) / 25
    offset = 25
    for i in range(batches):
        yield i
        i += offset
        
        
def getBatch(batch_size = 25, look_ahead = 1):
    image_batch = []
    batch_targets = []
    batch_len = batch_size   
    batches = len(datasets[0]) / batch_len
    offset = 0
    for k in range(batches):
        
        for i in range(batch_len):
            if i % 50 == 0:
                print(i)
            index = i + offset
            image, target = getImageArray(first_index=index)
            image_batch.append(image)
            batch_targets.append(target)
        
        batch_array = np.array(image_batch, dtype=np.float32)
        target_array = np.array([batch_targets], dtype=np.float32)
        target_array = target_array.transpose(1,0)
        
        offset += batch_len
        return batch_array, target_array
        #yield batch_array, target_array
  
    
def fullTrain():
    counter = 0
    for batch_array, target_array in getBatch():
        #batch_array, target_array = getBatch()
        train(10, batch_array, target_array)
        
        if counter % 100 == 0:
            serializers.save_npz('Forexxer v.01', model)
        counter += 1
        
            
def createTrainingArrays(batch_size=1000):
    image_batch = []
    batch_targets = []
    batch_len = batch_size   
    batches = len(datasets[0]) / batch_len
    offset = 0
    for k in range(batches):
        image_batch = []
        batch_targets = [] 
       
        for i in range(batch_len):
            if i % 50 == 0:
                print(k, i)
            index = i + offset
            image, target = getImageArray(first_index=index)
            image_batch.append(image)
            batch_targets.append(target)
        
        batch_array = np.array(image_batch, dtype=np.float32)
        target_array = np.array([batch_targets], dtype=np.float32)
        target_array = target_array.transpose(1,0)
        
        offset += batch_len
        np.save("Training/training_array " + str(offset), batch_array)
        #return batch_array, target_array
    
        

#batch_array, target_array = getBatch(batch_size = 1000)
        
#send them to NP array and 
#serializers.load_npz('Forexer v0.0', model)
#if save == "yes": serializers.save_npz('NFL boy v.03softmax 04', model)
      
#regression_model, optimizer, model = makeModel()
 
#files, datasets = loadFilledFiles() 
#datasets = chopDatasets(datasets, 20000)
#createTrainingArrays()

#full_2d_colors_array, target_value = getImageArray(look_ahead=1, last_index=4000)
 
#batch_array, target_array = getBatch()

#fullTrain()

#serailizers.load_npz('Forexxer v.01', model)

createTargetArrays() 

    

#train(100, batch_array, target_array)


#image = image[np.newaxis, :, :, :]

#adjusted_target_value = target_value / (10 ** len(str(int(target_value))))

#loss = regression_model(image, np.array([[.054]], dtype=np.float32))
      
#train(100, image, np.array([[adjusted_target_value]], dtype=np.float32))

#next we need a function that generates random sections of image and presents it in a batch


"""showing our imges"""
#full_2d_colors_array = full_2d_colors_array.transpose(1,2,0)
#x = Image.fromarray(np.uint8(full_2d_colors_array))
#x.show()
#x.save('out.png')














