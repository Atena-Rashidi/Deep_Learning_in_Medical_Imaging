""" @author Atena """
''' RGB to Gray_Scale '''

''' Here we want to convert RGB images to Gray_Scale. We define a class and 
call it from Data_Preprocessing.py. 

First we creat _init__ method or constructor to initializing the object’s state.
The task of constructors is to initialize (assign values) to the data members of 
the class when an object of the class is created

We creat a blank image using np.zero with the same shape of the initial image 
and then use cv.cvtColor(img, cv.COLOR_BGR2GRAYA) function to cover the images to gray scale and store them
on the blank image. '''

import numpy as np
from skimage.color import rgb2gray
import cv2 as cv

class Pre_Processing_RGB2GRAY():
    def __init__(self, image):
        self.image = image
    
    def R_2_G(self):
        Image_Number = self.image.shape[0]      # Batch_Size
        Image_Height = self.image.shape[1]      # Rows
        Image_Width = self.image.shape[2]       # Colomns

        Image_R_2_G = np.zeros((Image_Number, Image_Height, Image_Width), 
                                        dtype = np.uint8)

        for i in range(Image_Number):
           Image_R_2_G[i] = cv.cvtColor(self.image[i], cv.COLOR_RGB2GRAY)
           #Image_R_2_G[i] = rgb2gray(self.image[i]) 
        
        return Image_R_2_G           # return should be out of for to have all the images 

    

