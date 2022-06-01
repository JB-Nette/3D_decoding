import os

from utils import register_translation
from tifffile import imread, imsave
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
import glob

hyb_path = 'Y:/Jeeranan_analysis_KMS/0418/hyb'
image_name = 'hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F114.tif'

seg_path = 'Y:/Jeeranan_analysis_KMS/0418/stain'
dapi_img_list = os.listdir(seg_path)
dapi_seg_list = []
for img in dapi_img_list:
    if 'starfinity' in img:
        dapi_seg_list.append(img)
dapi_seg_list.sort()

image_hyb = imread(os.path.join(hyb_path , image_name))

