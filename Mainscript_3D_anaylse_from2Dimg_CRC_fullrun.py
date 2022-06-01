import os
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
from tifffile import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from utils import *
from PIL import Image

#This is 3D decoding that will generate .csv file of gene name, x, y, and z coordinates
crop_img = True

main_path = "C:/Users/synbio/Downloads/3D_seg/"
output_path = "C:/Users/synbio/Downloads/3D_seg/analysez=37_2D_fullfov"
img_hyb_0 = imread(main_path + "Ch1-sevengene.tif") # 3D array
img_hyb_1 = imread(main_path + "Ch2-sevengene.tif") # 3D array
img_hyb_2 = imread(main_path + "Ch3-sevengene.tif") # 3D array
img_hyb_3 = imread(main_path + "Ch4-sevengene.tif") # 3D array
img_hyb_4 = imread(main_path + "Ch5-sevengene.tif") # 3D array
img_hyb_5 = imread(main_path + "Ch6-sevengene.tif") # 3D array
img_hyb_6 = imread(main_path + "Ch7-sevengene.tif") # 3D array


n_stack = img_hyb_1.shape[0] # total of z stack in hyb image
ref_chan = 4
z_ref = n_stack//2
print(z_ref)
img_hyb_ref_atz = img_hyb_4[z_ref,:,:]
img_list =  [img_hyb_0[z_ref,:,:], img_hyb_1[z_ref,:,:], img_hyb_2[z_ref,:,:], img_hyb_3[z_ref,:,:], img_hyb_4[z_ref,:,:], img_hyb_5[z_ref,:,:], img_hyb_6[z_ref,:,:]]

# One plane register
# Register DAPI to ref channel
dapi_image_file = "C:/Users/synbio/Downloads/3D_seg/dapi_membrane.tif"
img_dapi = imread(dapi_image_file)
img_dapi_atz = img_dapi[z_ref,1, :,:] # 2D image (xdim, ydim)  at ref chan and ref z
register_dapi, _ = register_slice(img_hyb_ref_atz, img_dapi_atz)
#imsave(output_path +  "registered_dapi_1z" +".tif", register_dapi)

#crop image
if crop_img == True:
    crop_start = 0
    crop_end = 2000
    dapi_crop = register_dapi[crop_start:crop_end ,crop_start:crop_end]
    imsave(output_path + '/dapi_crop.tif', dapi_crop)
    import cv2
    dapi_crop.astype(np.uint8)
    cv2.imwrite(output_path + '/dapi_crop.png', dapi_crop)
    fig, axes = plt.subplots(nrows=2, ncols=4)
    ax = axes.ravel()
    for i in range(0,7):
        print(i)
        ax[i].imshow(img_list[i][crop_start:crop_end ,crop_start:crop_end], vmax =np.percentile(img_list[i],99.9),  cmap = 'gray')
        ax[i].axis('off')
    ax[7].imshow(dapi_crop [:,:], cmap='gray')
    ax[7].axis('off')
    plt.savefig(output_path + '/all_genes_and_dapi_crop.tif')



ref_slice = img_hyb_ref_atz[crop_start:crop_end ,crop_start:crop_end]
filter_param = [200,None] #[low cut, high cut] # we does not use this for 3D
threshold_mode = 'abs' #rel (relative value) or abs (absolute value)
dpi=200 #resolution for save image
x_list = []
y_list = []
z_list = []
gene_list = []

#threshold_list = [0.02,0.01,0.02,0.01,0.03,0.01,0.01]
percentiile_list = [99.95,99.94,99.94,99.93,99.95,99.96,99.95]
coord_img_list = []

for chan in range(0, 7):
    print("analysing channel number", chan)
    #data_index_list = []
    hyb_image_arr = img_list[chan][crop_start:crop_end ,crop_start:crop_end]
    register_image , shift = register_slice(ref_slice, hyb_image_arr)
    imsave(output_path + '/registered_img_channel_' + str(chan) + ".tif", register_image)
    image_norm = norm_image_func(register_image)
    filter_image = filter_func(filter_param, image_norm)
    percent = percentiile_list[chan]
    threshold = np.percentile(filter_image, percent)
    #threshold = threshold_list[chan]
    print('threshold is ', threshold)
    gene_coordinates_2D = find_peak_local_max(filter_image, threshold, threshold_mode)
    #all_z_list = np.unique(sort_z_plane_from_3D_coor[:, 0])
    #for z in z_choose_list:
    #data_index = [i for i, x in enumerate(sort_z_plane_from_3D_coor[:, 0] == z) if x]
    #data_index_list = data_index_list + data_index
    #sort_z_plane_from_3D_coor = sort_z_plane_from_3D_coor[data_index_list]

    # To register every channel to 1 ref channel and 1 ref z before we shift all spots to that
    #gene_coordinates_2D[:,0] = gene_coordinates_2D[:,0] + shift[1]
    #gene_coordinates_2D[:,1] = gene_coordinates_2D[:,1] + shift[0]
    # Visualize spot for 3 planes
    #coor_same_z = sort_z_plane_from_3D_coor[sort_z_plane_from_3D_coor[:,0] == z_ref]
    plt.figure(figsize = (15,15))
    plt.imshow(register_image, cmap='gray', vmax =np.percentile(register_image,99.9))
    plt.plot(gene_coordinates_2D[:,1], gene_coordinates_2D[:,0], 'ro', markersize=2)
    plt.axis('off')
    plt.savefig(output_path + '/' + 'plot1z_channel_' + str(chan) + '.png', dpi=500)
    plt.close()
    #plot_spot_in_3D_overlaydapi(hyb_image_arr, dapi_image_arr, sort_z_plane_from_3D_coor, output_path, dpi, alpha)
    # save spot
    #gene_name = np.array(['gene_' + str(chan)] * len(sort_z_plane_from_3D_coor)).reshape(-1,1)
    #coord_and_gene = np.hstack((sort_z_plane_from_3D_coor, gene_name))
    gene_name_list = ['gene_' + str(chan)] * len(gene_coordinates_2D[:,1])
    x_list = x_list + list(gene_coordinates_2D[:,0])
    y_list = y_list + list(gene_coordinates_2D[:,1])
    gene_list = gene_list + gene_name_list
data = [z_list, x_list, y_list, gene_list]
df = pd.DataFrame(data)
new_df = df.T
df_tosave = new_df.set_axis([ 'z', 'y', 'x', 'gene'], axis=1, inplace=False)
df_tosave.to_csv(output_path +  "/savespot_2D_1.csv")

# for i in range(0,7):
#     image = hyb_image_arr_used_atz[i,:,:]
#     imsave(output_path +  "crop_image_atz= " + str(z_ref) + "_chan_" +str(i) +".tif", image)

# Show crop image after register and plus DAPI (1 z)


