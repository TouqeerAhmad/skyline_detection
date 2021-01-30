import numpy as np
import cv2


class learnedFilters:
    def __init__(self, image, gray, target, coBins = 3, srBins = 6, orBins = 16, filterSz = 7):
        
        self.image = np.array(image, dtype=np.float32)
        self.gray = np.array(gray, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.shape = image.shape
        
        self.coBins = coBins
        self.srBins = srBins
        self.orBins = orBins
        self.filterSz = filterSz
        
        
        self.coherence = np.zeros(shape=(image.shape[0],image.shape[1]), dtype=np.float32)
        self.strength = np.zeros(shape=(image.shape[0],image.shape[1]), dtype=np.float32)
        self.orientation = np.zeros(shape=(image.shape[0],image.shape[1]), dtype=np.float32)
        
        
        self.coArr =  np.zeros(shape=(image.shape[0],image.shape[1]))
        self.srArr =  np.zeros(shape=(image.shape[0],image.shape[1]))                               
        self.orArr =  np.zeros(shape=(image.shape[0],image.shape[1]))
        
        
        self._structureTensor()
        self._structureTensorToIndex()
        
        return
    
    
    
    def _derivatives(self, chIndex):
    
    
        hor_kernel = np.array([[0.0, 0.70711], [-0.70711, 0.0]])
        ver_kernel = np.array([ [-0.70711, 0.0], [0.0, 0.70711]])
    
        image_x = cv2.filter2D(self.image[:,:,chIndex], -1, hor_kernel, borderType=cv2.BORDER_REPLICATE)
        image_y = cv2.filter2D(self.image[:,:,chIndex], -1, ver_kernel, borderType=cv2.BORDER_REPLICATE)
    
    
        image_xx = np.multiply(image_x,image_x)
        image_xy = np.multiply(image_x,image_y)
        image_yy = np.multiply(image_y,image_y)
    
        kernel = np.array([0.0053190,0.0947538,0.3999272,0.3999272,0.0947538,0.0053190])
        anchorPoint = (2,2)
    
        image_xx = cv2.sepFilter2D(image_xx,cv2.CV_32F,kernel,kernel,anchor=anchorPoint,borderType=cv2.BORDER_REPLICATE)
        image_xy = cv2.sepFilter2D(image_xy,cv2.CV_32F,kernel,kernel,anchor=anchorPoint,borderType=cv2.BORDER_REPLICATE)
        image_yy = cv2.sepFilter2D(image_yy,cv2.CV_32F,kernel,kernel,anchor=anchorPoint,borderType=cv2.BORDER_REPLICATE)
    
    
        return image_xx, image_xy, image_yy
    
    
    def _structureTensor(self):
    
        B_xx, B_xy, B_yy = self._derivatives(0)
        G_xx, G_xy, G_yy = self._derivatives(1)
        R_xx, R_xy, R_yy = self._derivatives(2)
        
        sum_xx = B_xx + G_xx + R_xx
        sum_xy = B_xy + G_xy + R_xy
        sum_yy = B_yy + G_yy + R_yy
        
    
        delta = np.sqrt(np.square(sum_xx - sum_yy)  + np.square(2 * sum_xy)) 
    
        lambda1 = (sum_xx + sum_yy + delta) / 2.0
        lambda2 = (sum_xx + sum_yy - delta) / 2.0
        
        lambda1[lambda1 < 0.0] = 0.0
        lambda2[lambda2 < 0.0] = 0.0
    
    
        strength = np.sqrt(lambda1)
        coherence = (np.sqrt(lambda1) - np.sqrt(lambda2)) / (np.sqrt(lambda1) + np.sqrt(lambda2))
    
        coherence = np.nan_to_num(coherence)
    
    
        w11 = 2.0 * sum_xy
        w12 = sum_yy - sum_xx + delta
    
    
    
        temp_w11 = w11
        w11 = w11 - w12
        w12 = w12 + temp_w11

    
        orientation = 180.0 * np.arctan(w12 / w11) / 3.1415
    
        if np.min(orientation) < 0.0:
            orientation += np.abs(np.min(orientation))
    
        #print(np.max(strength))
        #print(np.min(strength))
        
        #window_name = 'Sr'
        #cv2.imshow(window_name, normalize(strength))
        #cv2.waitKey()
        
        #window_name = 'Co'
        #cv2.imshow(window_name, normalize(coherence))
        #cv2.waitKey()
        
        #window_name = 'Or'
        #cv2.imshow(window_name, normalize(orientation))
        #cv2.waitKey()
    
        self.strength = strength 
        self.coherence = coherence 
        self.orientation = orientation
        
        return 
    
    
    def _structureTensorToIndex(self):
        #maxSr = 120.0
        maxSr = 180.0
        maxCo = 0.8
        maxOr = 180.0
    
        
        srArr = np.clip(self.strength - 40.0, 0.0, maxSr)
        coArr = np.clip(self.coherence, 0.0, maxCo)
        orArr = np.clip(self.orientation, 0.0, maxOr)
    
        srInterval = maxSr / (self.srBins -1)
        coInterval = maxCo / (self.coBins -1)
        orInterval = maxOr / (self.orBins)
    
    
    
        srArr = np.floor(srArr / srInterval)
        srArr = np.array(srArr, dtype=np.uint8)
    
    
        coArr = np.floor(coArr / coInterval)
        coArr = np.array(coArr, dtype=np.uint8)
    
    
        orShift = orInterval / 2.0
        orArr -= orShift
    
        orArr = np.floor(orArr / orInterval)
        orArr[orArr < 0.0] = 0.0
        orArr = np.array(orArr, dtype=np.uint8)
    
    
        self.srArr = srArr
        self.coArr = coArr
        self.orArr = orArr
    
        return
    
    def updateGramMatrix(self, G, M):
        
        H, W, Ch = self.shape
    
        halfFilterSz = int(np.floor(self.filterSz / 2.0))
    
        imgPatchVect = np.zeros(shape=(self.filterSz*self.filterSz+1,1), dtype=np.float32)
    
    
        for row in range(halfFilterSz,H-halfFilterSz):
            for col in range(halfFilterSz,W-halfFilterSz):
                if self.target[row,col] > 0.0:
                    srIndex = self.srArr[row,col]
                    coIndex = self.coArr[row,col]
                    orIndex = self.orArr[row,col]
                    
                    imgPatch = self.gray[row-halfFilterSz:row+halfFilterSz+1, col-halfFilterSz:col+halfFilterSz+1]
                    imgPatchVect[0:self.filterSz*self.filterSz,:] = np.reshape(imgPatch,newshape=(self.filterSz*self.filterSz,1))
                    imgPatchVect[self.filterSz*self.filterSz,:] = self.target[row,col]
                    
                    G[coIndex,srIndex,orIndex,:,:] = G[coIndex,srIndex,orIndex,:,:] + np.outer(imgPatchVect,imgPatchVect.T)
                    M[coIndex,srIndex,orIndex] = M[coIndex,srIndex,orIndex] + 1 
            
        return G, M
    
    
    def generateOutput(self, filterSet):
        
        H, W, Ch = self.shape    
        halfFilterSz = int(np.floor(self.filterSz / 2.0))
        
        imgPatchVect = np.zeros(shape=(self.filterSz*self.filterSz,1), dtype=np.float32)
        output = np.zeros(shape=(H,W), dtype=np.float32)
        
        for row in range(halfFilterSz,H-halfFilterSz):
            for col in range(halfFilterSz,W-halfFilterSz):
                if self.target[row,col] > 0:
                    imgPatch = self.gray[row-halfFilterSz:row+halfFilterSz+1, col-halfFilterSz:col+halfFilterSz+1]
                    imgPatchVect[0:self.filterSz*self.filterSz,:] = np.reshape(imgPatch,newshape=(self.filterSz*self.filterSz,1))
                    
                    coIndex,srIndex,orIndex = self.coArr[row,col], self.srArr[row,col],  self.orArr[row,col]
                    
                    currentFilter = filterSet[coIndex,srIndex,orIndex,:]
                    
                    output[row,col] = np.dot(currentFilter.T, imgPatchVect)
        
        return output
    