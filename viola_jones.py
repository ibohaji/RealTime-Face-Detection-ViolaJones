import numpy as np 
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import integral_image,resize
import unittest
from RectangleRegion import* 
from HaarFeature import* 


def compute_integral(img):

    return np.cumsum(np.cumsum(img, axis=0), axis=1)


class viola_jones:
    def __init__(self,T):
        self.T = T 

    def compute_haar_feature(self,imgs,y):
        """ input: array of variance normalized images, vector of labels for each img 
            output: Array of all features with the corresponding label value
        """
 
        labels = []
        all_features = []
        for idx,img in enumerate(imgs):
            print("computing img nr {} out of {}".format(idx,len(y)))
            features = []
            ii = compute_integral(img)
            height,width = img.shape  
   
            for i in range(0,height):
                for j in range(0,width):
                    h = 1
                    while(i+h-1<height):
                        w = 1 
                        while(j+w-1<width):

                            if(j+2*w-1<width):
                                left = RectangleRegion(i,j,w,h)
                                right = RectangleRegion(i,j+w,w,h)
                                value = right.compute_feature(ii) - left.compute_feature(ii)
                                feature = HaarFeature(1,i,j,w,h,value)
                                features.append(feature)
                                
                            if(i+2*h-1<height):
                                top = RectangleRegion(i,j,w,h)
                                bottom = RectangleRegion(i+h,j,w,h)
                                value = bottom.compute_feature(ii) - top.compute_feature(ii)
                                feature = HaarFeature(2,i,j,w,h,value)

                                features.append(feature)

                            if(j+3*w-1<width):
                                left = RectangleRegion(i,j,w,h)
                                mid = RectangleRegion(i,j+w,w,h)
                                right = RectangleRegion(i,j+2*w,w,h)
                                value = mid.compute_feature(ii) - (left.compute_feature(ii)+right.compute_feature(ii))
                                feature = HaarFeature(3,i,j,w,h,value)

                                features.append(feature)

                            if(i+3*h-1<height):
                                top = RectangleRegion(i,j,w,h)
                                mid = RectangleRegion(i+h,j,w,h)
                                bottom = RectangleRegion(i+2*h,j,w,h)
                                value = mid.compute_feature(ii) - (top.compute_feature(ii)+bottom.compute_feature(ii))
                                feature = HaarFeature(3,i,j,w,h,value)

                                features.append(feature)

                            if((i+2*h-1<height) & (j+2*w-1<width)):
                                tl = RectangleRegion(i,j,w,h)
                                tr = RectangleRegion(i,j+w,w,h)
                                bl = RectangleRegion(i+h,j,w,h)
                                br = RectangleRegion(i+h,j+w,w,h)

                                value = (tr.compute_feature(ii)+bl.compute_feature(ii)) - (tl.compute_feature(ii) + br.compute_feature(ii))
                                feature = HaarFeature(4,i,j,w,h,value)
                                features.append(feature)
                            w+=1
                        h+=1 

            print("img nr: {} computed".format(idx))
            all_features.append(features)
            labels.append(y[idx])


        return all_features,np.array(labels)   # Returns the features and corresponidng idx
