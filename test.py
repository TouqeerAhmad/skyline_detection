import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
import os
import scipy.io as sio

from learned_filters import learnedFilters as LRFS

import time

def normalize(image):
    #print(np.min(image), np.max(image))
    
    minVal = np.min(image)
    
    if minVal < 0.0:
        image += np.abs(minVal)
    
    maxVal = np.max(image)
    image /= maxVal
    
    return image



def processSingleImage(count, fileName, allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, flagSaveVisuals):
    
    
    filterSzHalf = int(filterSz/2.0)
                
    image_org = cv2.imread(fileName)
    image = cv2.copyMakeBorder(image_org,filterSzHalf,filterSzHalf,filterSzHalf,filterSzHalf,cv2.BORDER_REFLECT)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_mask = cv2.Canny(np.array(image_gray, dtype=np.uint8), 50.0,220.0)
    
        
    # intermediate output for debugging/visualization
    if flagSaveVisuals:
        cv2.imwrite(dirForImageOutput + str(count) + '_0.png',np.array(image, dtype=np.uint8))
        cv2.imwrite(dirForImageOutput + str(count) + '_1.png',np.array(canny_mask, dtype=np.uint8))
    
        
    lrfObj = LRFS(image, image_gray, canny_mask, coBins, srBins, orBins, filterSz)
    mask = lrfObj.generateOutput(allFilters)

        
    # gradient strength -- computed as part of ST
    strImage = lrfObj.strength.copy()
    strImage = normalize(strImage)
    
    
    mask[mask < 0.0] = 0.0
    mask[mask > 255.0] = 255.0

    
    # intermediate output for debugging/visualization
    if flagSaveVisuals:
        maskRGB = generateHeatMap(mask)
        cv2.imwrite(dirForImageOutput + str(count) + '_2.png',np.array(maskRGB, dtype=np.uint8))
        strImage_orgSz = strImage.copy()
        strImage_orgSz *= 255.0
        cv2.imwrite(dirForImageOutput + str(count) + '_3.png',np.array(strImage_orgSz, dtype=np.uint8))
        
    
    # saving the original size DCSI on which need to run shortest path
    maskDCSI = generateDenseMap(mask,strImage)
    H, W = maskDCSI.shape
    maskDCSI_orgSz = maskDCSI[filterSzHalf:H-filterSzHalf, filterSzHalf:W-filterSzHalf].copy()
    sio.savemat(dirForMatFiles + str(count) + '.mat', {'maskDCSI_orgSz':maskDCSI_orgSz})
    
    
    # intermediate output for debugging/visualization
    if flagSaveVisuals:
        maskDCSI /= 2.0
        maskDCSI *= 255.0
        cv2.imwrite(dirForImageOutput + str(count) + '_4.png',np.array(maskDCSI, dtype=np.uint8))

    return
    


def runInference(allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, datasetName, flagSaveVisuals):
    
    if datasetName == 'Basalt':
        totalTime = 0.0
        numImages = 45    
        for imgIndex in range(1,numImages+1):
            print(imgIndex)
            
            if imgIndex < 10:
                fileName = './data/Basalt/images/marsim000' + str(imgIndex) + '.pgm'
            else:
                fileName = './data/Basalt/images/marsim00' + str(imgIndex) + '.pgm'
                
            start = time.time()
            processSingleImage(imgIndex, fileName, allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, flagSaveVisuals)
            end = time.time()
            totalTime += (end - start)
            
        print('Total Time: ', totalTime)
        print('Average Time: ', totalTime/numImages)
        
    elif datasetName == 'Web':
        totalTime = 0.0
        numImages = 80    
        for imgIndex in range(1,numImages+1):
            print(imgIndex)
            
            if imgIndex < 10:
                fileName = './data/web_dataset/images/R_GImag000' + str(imgIndex) + '.bmp'
            else:
                fileName = './data/web_dataset/images/R_GImag00' + str(imgIndex) + '.bmp'
            start = time.time()
            processSingleImage(imgIndex, fileName, allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, flagSaveVisuals)
            end = time.time()
            totalTime += (end - start)
            
        print('Total Time: ', totalTime)
        print('Average Time: ', totalTime/numImages)
        
    elif datasetName == 'CH1':
        totalTime = 0.0
        numImages = 203
        imgIndex = 1
        basePathNameList = ['./data/CH1/panoramio/images', './data/CH1/cvg/images', './data/CH1/poor_edge_images/images']
        
        for index in range(0,3):
            basePathName = basePathNameList[index]
            listOfFiles = listdir(basePathName)
            for fName in listOfFiles:
                fileName = os.path.join(basePathName, fName)
                print(imgIndex, fileName)
                
                start = time.time()
                processSingleImage(imgIndex, fileName, allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, flagSaveVisuals)
                end = time.time()
                totalTime += (end - start)
                imgIndex += 1
                
        print('Total Time: ', totalTime)
        print('Average Time: ', totalTime/numImages)
        
    else:
         # for GeoPose3K dataset
        totalTime = 0.0
        numImages = 2895
        mat_contents = sio.loadmat('./misc/NamesOf2895Files.mat')
        basePathName = './data/geoPose3K_rescaled/'
        
        
        for imgIndex in range(0,numImages):
            dirName = os.path.join(basePathName,str(mat_contents["B"][imgIndex][0][0][0]))
            fileName = os.path.join(dirName,"photo.jpg")
            
            print(imgIndex+1, fileName)
            
            start = time.time()
            processSingleImage(imgIndex+1, fileName, allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, flagSaveVisuals)
            end = time.time()
            totalTime += (end - start)
            
        print('Total Time: ', totalTime)
        print('Average Time: ', totalTime/numImages)
    
        
    return




def generateDenseMap(image, srImage):
    
    H, W = image.shape
    hMap = 255.0 * np.zeros(shape=(H,W), dtype=np.float32)
    
    for row in range(0,H):
        for col in range(0,W):
            tempVal = 0.5 * (1.0 - image[row,col] / 255.0) + 0.5 * (1.0 - srImage[row,col])
            hMap[row,col] = tempVal
                    
    return hMap



def generateHeatMap(image):
    
    H, W = image.shape
    hMap = 255.0 * np.zeros(shape=(H,W,3), dtype=np.float32)
    
    for row in range(0,H):
        for col in range(0,W):
            if image[row,col] > 0.0:
                tempVal = image[row,col] / 255.0 - 0.5
                if tempVal <= 0.0:
                    hMap[row,col,0] = np.abs(tempVal+0.5) * 510.0
                    hMap[row,col,1] = 0.0
                    hMap[row,col,2] = 0.0                    
                else:
                    hMap[row,col,2] = np.abs(tempVal) * 510.0
                    hMap[row,col,0] = 0.0
                    hMap[row,col,1] = 0.0
    
    return hMap



def main():
    
    srBins = 6
    coBins = 3
    orBins = 16
    filterSz = 7 #9 #11 #13 #15
    
    datasetName_Filter = 'Basalt' #'Basalt' #'Web' #'CH1'
    datasetName_Inference = 'Web' #'Basalt' #'Web' #'CH1' #'GeoPose3K'
    
    fileNameFilters = './filterBank/' +  datasetName_Filter + '_filterSz_' + str(filterSz) + '_filters.npy'
    allFilters = np.load(fileNameFilters)
    
    dirForImageOutput = './output/images_for_' + datasetName_Inference + '_FilterSz_7_trained_on_' + datasetName_Filter + '/' 
    dirForMatFiles = './output/mats_for_' + datasetName_Inference + '_FilterSz_7_trained_on_' + datasetName_Filter + '/'
    
    if not os.path.exists(dirForImageOutput):
        os.makedirs(dirForImageOutput)
    if not os.path.exists(dirForMatFiles):
        os.makedirs(dirForMatFiles)
    
    runInference(allFilters, dirForImageOutput, dirForMatFiles, srBins, coBins, orBins, filterSz, datasetName_Inference, 1)
    
    return


if __name__ == '__main__':
    main()
