import numpy as np 
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import integral_image,resize
import unittest

class HaarFeature: 
    def __init__(self,type,y,x,width,height,value):
        self.type = type 
        self.x = x 
        self.y = y 
        self.width = width 
        self.height = height 
        self.value = value 