#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:50:59 2021

@author: touqeerahmad
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
import os
import scipy.io as sio
import time


filterSz = 16


def normalize(image):
    
    minVal = np.min(image)
    
    if minVal < 0.0:
        image += np.abs(minVal)
    maxVal = np.max(image)
    image /= maxVal
    
    return image


def runInference(matContent, dirForImageOutput, dirForMatFiles):
    
    
    svmWeights = matContent['A'][:,0]
    totalTime = 0.0
    
    # for web dataset    
    numImages = 80
    for imgIndex in range(1,numImages+1):
        print(imgIndex)
        
        if imgIndex < 10:
            fileName = './data/web_dataset/images/R_GImag000' + str(imgIndex) + '.bmp'
        else:
            fileName = './data/web_dataset/images/R_GImag00' + str(imgIndex) + '.bmp'
            
    
        start = time.time()

    
        filterSzHalf = int(filterSz/2.0)
        
        image_org = cv2.imread(fileName)
        image = cv2.copyMakeBorder(image_org,filterSzHalf,filterSzHalf,filterSzHalf,filterSzHalf,cv2.BORDER_REFLECT)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.array(image_gray, dtype=np.float32)
        
        image_gray -= 128.0
        image_gray /= 128.0
        
        
        
        H, W = image_gray.shape
        maskDCSI = 255.0 * np.zeros(shape=(H,W), dtype=np.float32)
        
        for row in range(9,H-9):
            for col in range(9,W-9):
                block = image_gray[row-7:row+9,col-7:col+9]
                
                dot_product = np.dot(np.reshape(block, (1,256), order='F'), np.expand_dims(svmWeights[0:256],axis=1))
                dot_product += svmWeights[256] 
                maskDCSI[row,col] = dot_product
                
        
        maskDCSI_orgSz = maskDCSI.copy()
        maskDCSI_orgSz = 1.0 - normalize(maskDCSI_orgSz)
        sio.savemat(dirForMatFiles + str(imgIndex) + '.mat', {'maskDCSI_orgSz':maskDCSI_orgSz})
        
        
        maskDCSI_orgSz *= 255.0
        cv2.imwrite(dirForImageOutput + str(imgIndex) + '_0.png',np.array(maskDCSI_orgSz, dtype=np.uint8))
        
        end = time.time()
        
        totalTime += (end - start)
        
        
    print('Total Time: ', totalTime)
    print('Average Time: ', totalTime/numImages)
    
    return



if __name__ == '__main__':
    
    
    fileNameWeights = './misc/Horizon_SVM_Classifier_Baatz_CH1.mat'
    mat_contents = sio.loadmat(fileNameWeights)
    
    
    dirForImageOutput = './output/Web_horizon_DCSI_images/'
    dirForMatFiles = './output/Web_horizon_DCSI_mats/'
    
    if not os.path.exists(dirForImageOutput):
        os.makedirs(dirForImageOutput)
    if not os.path.exists(dirForMatFiles):
        os.makedirs(dirForMatFiles)
    
    runInference(mat_contents, dirForImageOutput, dirForMatFiles)
        
    