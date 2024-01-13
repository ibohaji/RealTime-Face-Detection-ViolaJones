import cv2
import math 


def downsample_square_img(img):
    """ 
    Downsample a square image to a dimension 24x24 using gaussian blur as a smoothing function 

    parameters: 
    - img: Input image (NumPy Array) 
    
    Returns: 
    -Downsampled 24x24 image 
    """ 

    e = img.shape[0]
    sigma = 0.6*math.sqrt((e/24)**2-1)
    blurred_image = cv2.GaussianBlur(img,(0,0),sigma)
    downsampled_img = cv2.resize(blurred_image,(24,24))

    return downsampled_img

def collect_false_positives()
