
# 4D tensor for image: (Batch-Size, height, width, Channels)
# image = np.arange(0, 50, 10, dtype=np.uint8)
# _________________________________________________________________________________
# Image data types:
# In skimage, images are simply numpy arrays, which support a variety of data types
# Data type	        Range
# ___________________________________
# uint8	            0 to 255
# uint16	        0 to 65535
# uint32	        0 to 232
# float	            0 to 1 or -1 to 1
# int8	            -128 to 127
# int16	            -32768 to 32767
# int32	            -2^31 to 2^31 - 1

''' Image Pre_processing Code Includes:

- One-Hot Encoding
- Image Re-sizeing
- Image Augmentation
- Image Recolor
- Image Saving
- Show Results 

'''

''' scikit-image automatically converts images to floating point any time that interpolation 
or convolution is necessary, to ensure precision in calculations. In converting to float, the 
range of the image is converted to [0, 1] '''

# 1. Import required modules
# 2. Set image path
# 3. Get images and set some parameters, One-Hot Encoding
# 4. Define funtions: Re_Size, Augmentation, Re_Color, Im_Saving
# 4.1. Function_1: Pre_Process_Re_Size():
# 4.2. Function_2: Pre_Process_Augmentation():
# 4.3. Function_3: Pre_Process_Re_Color():
# 4.4. Function_4: Pre_Process_Im_Saving(): 
# 5. Show Results
''' ___________________________________________________________________ '''
# 1. Import required modules
import os
import sys
import glob
import random
import numpy as np
import cv2 as cv 
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from keras_preprocessing import image
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage.viewer import ImageViewer
# Class - OPP file
from OPP import Pre_Processing_RGB2GRAY
''' ________________________________________________________________________ '''
# 2. Set image path
IMAGE_PATH = r"D:\\GitHub\\Deep_Learning_in_Medical_Imaging\\Breast_Sampling_Pathology_Images_Segmentation\\Dataset\\InputsTrain\\"
IMAGE_MASK_PATH = r"D:\\GitHub\\Deep_Learning_in_Medical_Imaging\\Breast_Sampling_Pathology_Images_Segmentation\\Dataset\\MasksTrain\\"

''' _________________________________________________________________________ '''
# 3. Get images and set some parameters, One-Hot Encoding
# we need to know the width and height of images
IMG_HEIGHT = 768
IMG_WIDTH = 896
IMG_CHANNELS = 3

''' Now, we need to have access to all the files in the defined path
we use a function from os: walk to access to these files
OS.walk() generate the file names in a directory tree by walking 
the tree either top-down or bottom-up.'''

IMG_Dataset = next(os.walk(IMAGE_PATH))[2]

# Now, we should read each image as a 4-D tensor 
# (Batch-Size, height, width, Channels, dtype)
# we generate two empty tensors for images and mask and fill 
# them out with images and mask

Inputs = np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                  dtype = np.uint8)
Ground_Truth = np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, 2), 
                    dtype = np.bool)  # channel is 2, because it white and black img

print('Looding Images & Masks...\nPlease Wait!')

''' Now, we want to extract and save the index and the name of file, existing in the 
IMG_Dataset, So, we use enumerate(), the function gives you back two loop variables:
1. The count of the current iteration
2. The value of the item at the current iteration '''


# tqdm is great for having a progress bar
# it necessary to add a sys.stdout.flush() call before usage to avoid possible interleaved output

sys.stdout.flush()
for n, f in tqdm(enumerate(IMG_Dataset), total = len(IMG_Dataset)):
    Images = imread (IMAGE_PATH + f)[:,:,:IMG_CHANNELS]
    Inputs[n] = Images

    Masks = imread(IMAGE_MASK_PATH + f.replace('_ccd.tif','.TIF'))
    Masks = np.squeeze(Masks).astype(np.bool) 

    ''' The squeeze() function is used to remove single-dimensional entries 
    from the shape of an array 
    here, we use this function if there is a black/which picture 
    with dimention = 1, to omit it and to change it with boolean values
     x = np.array([[[0], [1], [2]]])
        x.shape
        (1, 3, 1)
        np.squeeze(x).shape
        (3,) '''

    # One-Hot Encoding
    Ground_Truth[n, :, :, 0] = ~Masks
    Ground_Truth[n, :, :, 1] = Masks

print('\nLoading Images and Masks Completed Successfully!')

''' _________________________________________________________________________ '''
# 4. Define funtions: Re_Size, Augmentation, Re_Color, Im_Saving
# 4.1. Function_1: Pre_Process_Re_Size():

IMG_HEIGHT_RESIZED = 225
IMG_WIDTH_RESIZED = 225
IMG_CHANNELS_RESIZED = 3

# imgs.shape[0] --> number of images/Batch_Size

def Pre_Process_Re_Size(imgs):
    img_p = np.zeros((imgs.shape[0], IMG_HEIGHT_RESIZED, IMG_WIDTH_RESIZED, 
                        IMG_CHANNELS_RESIZED), dtype = np.uint8)
     
    for i in range(imgs.shape[0]):
        img_p[i] = resize(imgs[i], (IMG_HEIGHT_RESIZED, IMG_WIDTH_RESIZED, 
                        IMG_CHANNELS_RESIZED), preserve_range = True)
        
    return img_p        # return should be out of for to have all the images 

''' anti_aliasing: bool, optional: 
____________________________________
Whether to apply a Gaussian filter to smooth the image prior to 
down-scaling. It is crucial to filter when down-sampling the image 
to avoid aliasing artifacts. If input image data type is bool, 
no anti-aliasing is applied. 

preserve_range: bool, optional:
________________________________
Whether to keep the original range of values. Otherwise, the input image 
is converted according to the conventions of img_as_float. '''


''' _________________________________________________________________________ '''
# 4.2. Function_2: Pre_Process_Augmentation():
''' Keras ImageDataGenerator 
_______________________________
mage augmentation is a technique of applying different transformations to original 
images which results in multiple transformed copies of the same image . K
eras ImageDataGenerator class provides real-time data augmentation, meaning it generates 
augmented images on the fly while the model is still in the training stage. 

The ImageDataGenerator class has three methods flow(), flow_from_directory() and flow_from_dataframe()
to read the images from a big numpy array and folders containing images.'''

# Now we should define two path for the intial image, in the Augmentation forlder 
# and the augmented images. Then we apply "image.ImageDataGenerator" to augment the
# images. We use flow_from_directory() to read initial images, and save the augmented ones

IMAGE_INITIAL_PATH = r"D:\\GitHub\\Deep_Learning_in_Medical_Imaging\\Breast_Sampling_Pathology_Images_Segmentation\\Dataset\\Augmentation\\"
IMAGE_AUGMENTATED_PATH = r"D:\\GitHub\\Deep_Learning_in_Medical_Imaging\\Breast_Sampling_Pathology_Images_Segmentation\\Dataset\\Rotated\\"

Date_Gen = image.ImageDataGenerator(rotation_range = 30) # you can augment it based on your desired parameter
IMG_AUG = Date_Gen.flow_from_directory(IMAGE_INITIAL_PATH, target_size=(768, 896), 
            batch_size=1, save_to_dir=IMAGE_AUGMENTATED_PATH, 
            save_prefix='Aug')
for i in range(8):
    IMG_AUG.next()

# In order to show the results, we define a function, showing all the images in a sub-plot

def Pre_Process_Augmentation(path_Image):
    Images_List = glob.glob(path_Image)            # to get access to this path
    Figure = plt.figure()

    for i in range(6):
        Images_A = Image.open(Images_List[i])       # another way of reading images like imread
        Sub_Image_Show = Figure.add_subplot(231+i)
        Sub_Image_Show.imshow(Images_A) 
    plt.show()
    return Figure
    
# Now, we can use this function to show the rotated (augmented) images and compare it with the original one

Image_Original = Pre_Process_Augmentation(IMAGE_INITIAL_PATH + 'input\\*')
Image_Original.savefig(IMAGE_AUGMENTATED_PATH + '\\Original.png', dpi = 200,
                                papertype = 'a5')

Image_Augmentation = Pre_Process_Augmentation(IMAGE_AUGMENTATED_PATH + '\\*')
Image_Augmentation.savefig(IMAGE_AUGMENTATED_PATH + '\\Rotated.png', dpi = 200,
                                papertype = 'a5')

# Now, we resize the images
Image_Resize = Pre_Process_Re_Size(Inputs)

''' _________________________________________________________________________ '''
# 4.3. Function_3: Pre_Process_Re_Color():
''' This function will be written using Object-Oriented-Programming (OOP) '''
# Please refer to the OOP.py file
# We should import the class from OPP in the module/libray section
# Now we define an object, using the class in OOP, we can creat an instance of the class.

Object_Pre_Processing = Pre_Processing_RGB2GRAY(Inputs)
# Now we creat an object/instance of the class and we can apply the method of the class on it.
# This Gray_Scale is a tensor and will be use for Pre_Process_Im_Saving
Gray_Scale = Pre_Processing_RGB2GRAY.R_2_G(Object_Pre_Processing)

''' _________________________________________________________________________ '''
# 4.4. Function_4: Pre_Process_Im_Saving(): 
New = r"D:\\GitHub\\Deep_Learning_in_Medical_Imaging\\Breast_Sampling_Pathology_Images_Segmentation\\Dataset\\GrayImages\\"

def Pre_Process_Im_Saving(path_Image, Path_Output, Tensor):
    for i, filename in enumerate(os.listdir(path_Image)):
        ''' imsave(fname, arr, plugin=None, check_contrast=True, **plugin_args) '''
        imsave(fname = '{}{}'.format(Path_Output, filename), 
                    arr = Tensor[i])
        print('{}: {}'.format(i, filename))

Pre_Process_Im_Saving(IMAGE_PATH, New, Gray_Scale)

''' _________________________________________________________________________ '''
# Show Result
ix = random.randint(0, len(Inputs))

img = Inputs[ix]
mask = Ground_Truth[ix]
resized = Image_Resize[ix]
gray = Gray_Scale[ix]

print("Input Image")
imshow(img)
plt.title('Input_Image')
plt.xlabel('Width_pix')
plt.ylabel('Height_pix')
plt.savefig('Input Image')
plt.show()


print('Mask')
imshow(mask[:, :, 0])
plt.title('Mask')
plt.xlabel('Width_pix')
plt.ylabel('Height_pix')
plt.savefig('Mask_Image')
plt.show()

print("Resized Image")
imshow(resized)
plt.title('Resized_Image')
plt.xlabel('Width_pix')
plt.ylabel('Height_pix')
plt.savefig('Resized_Image')
plt.show()

print('GrayScale Image')
imshow(gray)
plt.title('GrayScale Image')
plt.xlabel('Width_pix')
plt.ylabel('Height_pix')
plt.savefig('GrayScale_Image')
plt.show()
