import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
import os
import scipy.io as sio

from learned_filters import learnedFilters as LRFS


def makeGTMask(image, imageClean, filterSz):
    
    image = np.array(image, dtype=np.float32)
    H, W, C = image.shape
    
    halfFilterSz = int(np.floor(filterSz / 2.0))
    
    mask = cv2.Canny(np.array(imageClean, dtype=np.uint8), 50.0,220.0) #imageClean.copy()
    mask2 = np.zeros(shape=(H,W), dtype=np.float32)
    
    count = 0
    for col in range(W):
        for row in range(H):
           if (image[row,col,0] == 0 and image[row,col,1] == 0 and image[row,col,2] == 255.0) or (image[row,col,0] == 255.0 and image[row,col,1] == 0 and image[row,col,2] == 0.0):
               mask[row-halfFilterSz:row+halfFilterSz+1,col] = 0 * mask[row-halfFilterSz:row+halfFilterSz+1,col]
               mask2[row,col] = 255.0
               count += 1
               continue
    

    indices = mask > 0
    indicesTuple = np.where(indices == True)
    randNegativeLoc = np.random.randint(indicesTuple[0].shape[0], size=count)
    
    
    for k in randNegativeLoc:
        row, col = indicesTuple[0][k], indicesTuple[1][k]
        mask2[row,col] = 127.5
        
    
    return mask2


def makeGTForCH1(mask, image):
    
    gt = image.copy()
    H, W, C = gt.shape
    
    for col in range(W):
        for row in range(H-1):
           if mask[row,col,0] == 0 and mask[row+1,col,0] == 255:
               gt[row+1,col,0] = 0
               gt[row+1,col,1] = 0
               gt[row+1,col,2] = 255
               continue
               
    
    return gt
    


def processSingleImage(j, G, M, srBins, coBins, orBins, filterSz, image, image_ground_truth):
    
    if j == 1:
        image = cv2.flip(image,1)
        image_ground_truth = cv2.flip(image_ground_truth,1)
    elif j == 2:
        image = cv2.flip(image,0)
        image_ground_truth = cv2.flip(image_ground_truth,0)
    elif j == 3:
        image = cv2.flip(image,-1)
        image_ground_truth = cv2.flip(image_ground_truth,-1)
    elif j == 4:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_ground_truth = cv2.rotate(image_ground_truth, cv2.ROTATE_90_CLOCKWISE)
    elif j == 5:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_ground_truth = cv2.rotate(image_ground_truth, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image,0)
        image_ground_truth = cv2.flip(image_ground_truth,0)
    elif j == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_ground_truth = cv2.rotate(image_ground_truth, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image,1)
        image_ground_truth = cv2.flip(image_ground_truth,1)
    elif j == 7:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_ground_truth = cv2.rotate(image_ground_truth, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image,-1)
        image_ground_truth = cv2.flip(image_ground_truth,-1)                
    else:
        image = image
        image_ground_truth = image_ground_truth
                
            
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = np.array(image, dtype=np.float32)
    image_gray = np.array(image_gray, dtype=np.float32)
    image_ground_truth = np.array(image_ground_truth, dtype=np.float32)
    
    image_mask = makeGTMask(image_ground_truth, image_gray, filterSz)
    
    
    lrfObj = LRFS(image, image_gray, image_mask,  coBins, srBins, orBins, filterSz)
    G, M = lrfObj.updateGramMatrix(G, M)


    return G, M    


def processAllImages(G, M, srBins, coBins, orBins, filterSz, datasetName):
    
    
    if datasetName == 'Basalt':
        for augIndex in range(0,2):
            for imgIndex in range(1,46):
                print(augIndex,imgIndex)
                
                if imgIndex < 10:
                    fileName = './data/Basalt/images/marsim000' + str(imgIndex) + '.pgm'
                    fileNameGT = './data/Basalt/ground_truth/GT_marsim00' + str(imgIndex) + '_Edge.bmp'        
                else:
                    fileName = './data/Basalt/images/marsim00' + str(imgIndex) + '.pgm'
                    fileNameGT = './data/Basalt/ground_truth/GT_marsim0' + str(imgIndex) + '_Edge.bmp'
                
                print(fileName)
                image = cv2.imread(fileName)
                image_ground_truth = cv2.imread(fileNameGT)
                G, M = processSingleImage(augIndex, G, M, srBins, coBins, orBins, filterSz, image, image_ground_truth)
                
                
    elif datasetName == 'Web':
        for augIndex in range(0,2):
            for imgIndex in range(1,81):
                print(augIndex,imgIndex)
                if imgIndex < 10:
                    fileName = './data/web_dataset/images/R_GImag000' + str(imgIndex) + '.bmp'
                    fileNameGT = './data/web_dataset/ground_truth/GT_GoogleImage00' + str(imgIndex) + '_Edge.bmp'        
                else:
                    fileName = './data/web_dataset/images/R_GImag00' + str(imgIndex) + '.bmp'
                    fileNameGT = './data/web_dataset/ground_truth/GT_GoogleImage0' + str(imgIndex) + '_Edge.bmp'
                    
                print(fileName)
                image = cv2.imread(fileName)
                image_ground_truth = cv2.imread(fileNameGT)
                G, M = processSingleImage(augIndex, G, M, srBins, coBins, orBins, filterSz, image, image_ground_truth)
                
    else:
        # for CH1 dataset
        basePathNameList = ['./data/CH1/panoramio/images', './data/CH1/cvg/images']
        basePathNameGTList = ['./data/CH1/panoramio/ground_truth', './data/CH1/cvg/ground_truth']
        
        for index in range(0,2):
            basePathName = basePathNameList[index]
            basePathNameGT = basePathNameGTList[index]
        
            for j in range(0,2):
                listOfFiles = listdir(basePathName)
                for k in listOfFiles:
                    fileName = os.path.join(basePathName,k)
                    fileNameBase, file_extension = os.path.splitext(k)
                    k1 = fileNameBase + '-mask' + file_extension
                    fileNameGT = os.path.join(basePathNameGT,k1)
                    
                    
                    print(fileName)
                    image = cv2.imread(fileName)
                    image_ground_truth = makeGTForCH1(cv2.imread(fileNameGT), image)
                    G, M = processSingleImage(j, G, M, srBins, coBins, orBins, filterSz, image, image_ground_truth)
             
        
    return G, M




def runTraining(fileNameG, fileNameM, srBins, coBins, orBins, filterSz, datasetName):
    G = np.zeros(shape=(coBins,srBins,orBins,filterSz*filterSz+1,filterSz*filterSz+1), dtype=np.float32)
    M = np.zeros(shape=(coBins,srBins,orBins), dtype=np.float32)
    
    G, M = processAllImages(G, M, srBins, coBins, orBins, filterSz, datasetName)
    
    np.save(fileNameG, G)
    np.save(fileNameM, M)
    
    return


def solveRegressions(G, M, srBins, coBins, orBins, filterSz):
    
    Q = np.genfromtxt('./filterBank/Q_matrix.txt',delimiter=',')
    
    allFilters = np.zeros(shape=(coBins,srBins,orBins,filterSz*filterSz), dtype=np.float32)
    count = 0
    
    for coIndex in range(0,coBins):
        for srIndex in range(0,srBins):
            for orIndex in range(0,orBins):
                
                AtA = G[coIndex,srIndex,orIndex,0:filterSz*filterSz,0:filterSz*filterSz]
                Atb = G[coIndex,srIndex,orIndex,0:filterSz*filterSz,filterSz*filterSz]
                #btA = G[coIndex,srIndex,orIndex,49,0:49]
                #btb = G[coIndex,srIndex,orIndex,49,49]
                
                #filterVector = np.matmul(np.linalg.pinv(Q + AtA), Atb)
                filterVector = np.matmul(np.linalg.pinv(AtA), Atb)

                
                allFilters[coIndex,srIndex,orIndex,:] = filterVector
                #print(count, filterVector.shape)
                count += 1
    
    return allFilters

    

def main():
    
    srBins = 6
    coBins = 3
    orBins = 16
    filterSz = 7 #9 #11 #13 #15
    
    datasetName = 'Basalt' #'Web' #'CH1'

    
    fileNameG = './filterBank/' +  datasetName + '_filterSz_' + str(filterSz) + '_matrixG.npy'
    fileNameM = './filterBank/' +  datasetName + '_filterSz_' + str(filterSz) + ' _matrixM.npy'
    fileNameFilters = './filterBank/' +  datasetName + '_filterSz_' + str(filterSz) + '_filters.npy'
    
    
    runTraining(fileNameG,fileNameM, srBins, coBins, orBins, filterSz, datasetName)
    
    G = np.load(fileNameG)
    M = np.load(fileNameM)
    allFilters = solveRegressions(G, M, srBins, coBins, orBins, filterSz)
    
    #print(allFilters.shape)
    np.save(fileNameFilters, allFilters)
    
    return
    

if __name__ == '__main__':
    main()
    
    