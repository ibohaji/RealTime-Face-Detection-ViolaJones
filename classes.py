import numpy as np 
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import integral_image,resize
import unittest

#Computing integral image


def compute_integral(img):

    return np.cumsum(np.cumsum(img, axis=0), axis=1)


class RectangleRegion:

    def __init__(self,y,x,width,height):
        self.x = x 
        self.y = y 
        self.height = height 
        self.width = width 

    def compute_feature(self,ii):
        BR = ii[self.y-1+self.height-1,self.x+self.width-1]
        TL = ii[self.y-1,self.x-1] if self.y>0 and self.x>0 else 0 
        TR = ii[self.y-1,self.x+self.width-1] if self.y>0 else 0 
        BL = ii[self.y+self.height-1,self.x-1] if self.x>0 else 0 

        return BR + TL - TR-BL
    


class HaarFeature: 
    def __init__(self,type,y,x,width,height,value):
        self.type = type 
        self.x = x 
        self.y = y 
        self.width = width 
        self.height = height 
        self.value = value 



class viola_jones:
    def __init__(self,T):
        self.T = T 

    def compute_haar_feature(self,ii):

        features = []
        height,width = ii.shape  

        for i in range(0,height):
            for j in range(0,width):
                h = 1
                while(i+h-1<height):
                    w = 1 
                    while(j+w-1<width):

                        if(j+2*w-1<width):
                            left = RectangleRegion(i,j,w,h)
                            right = RectangleRegion(i,j+w,w,h)
                            feature = right.compute_feature(ii) - left.compute_feature(ii)
                            features.append(feature)
                            
                        if(i+2*h-1<height):
                            top = RectangleRegion(i,j,w,h)
                            bottom = RectangleRegion(i+h,j,w,h)
                            feature = bottom.compute_feature(ii) - top.compute_feature(ii)
                            features.append(feature)

                        if(j+3*w-1<width):
                            left = RectangleRegion(i,j,w,h)
                            mid = RectangleRegion(i,j+w,w,h)
                            right = RectangleRegion(i,j+2*w,w,h)
                            feature = mid.compute_feature(ii) - (left.compute_feature(ii)+right.compute_feature(ii))
                            features.append(feature)

                        if(i+3*h-1<height):
                            top = RectangleRegion(i,j,w,h)
                            mid = RectangleRegion(i+h,j,w,h)
                            bottom = RectangleRegion(i+2*h,j,w,h)
                            feature = mid.compute_feature(ii) - (top.compute_feature(ii)+bottom.compute_feature(ii))
                            features.append(feature)

                        if((i+2*h-1<height) & (j+2*w-1<width)):
                            tl = RectangleRegion(i,j,w,h)
                            tr = RectangleRegion(i,j+w,w,h)
                            bl = RectangleRegion(i+h,j,w,h)
                            br = RectangleRegion(i+h,j+w,w,h)

                            feature = (tr.compute_feature(ii)+bl.compute_feature(ii)) - (tl.compute_feature(ii) + br.compute_feature(ii))
                            features.append(feature)
                        w+=1
                    h+=1 
        return features      


    
#Test so far 

img = io.imread("DTUImageAnalysis/data/Fish/discus.jpg")
img = rgb2gray(img) 
img = resize(img,(24,24)) 


ii = compute_integral(img) 

viola = viola_jones(10)
from skimage.feature import haar_like_feature

features = viola.compute_haar_feature(ii) 
print(len(features))
print(len(features_1))


