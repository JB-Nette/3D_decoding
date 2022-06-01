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

#This is 3D decoding that will generate .csv file of gene name, x, y, and z coordinates
crop_img = True

hyb_image_file = "C:/Users/synbio/Downloads/3D_seg/sevengene.tif"
output_path = "C:/Users/synbio/Downloads/3D_seg/"
img_hyb = imread(hyb_image_file)

n_stack = img_hyb.shape[0] # total of z stack in hyb image
n_channels = img_hyb.shape[1] # total of channels in hyb image
hyb_image_arr_checked, _ = check_channels_and_output_multiplez(img_hyb, n_stack, n_channels)
z_start = 10
z_end = 65
ref_chan = 4
z_ref = ((z_start + z_end)//2)-z_start

# One plane register
# Register DAPI to ref channel
ref_z_dapi = ((z_start + z_end)//2)
dapi_image_file = "C:/Users/synbio/Downloads/3D_seg/dapi_membrane.tif"
img_dapi = imread(dapi_image_file)
img_dapi_atz = img_dapi[ref_z_dapi ,1 , :,: ]
img_hyb_ref  = hyb_image_arr_checked[ref_chan,ref_z_dapi,:,:]
register_dapi, _ = register_slice(img_hyb_ref, img_dapi_atz)
#imsave(output_path +  "registered_dapi_1z" +".tif", register_dapi)

#crop image
if crop_img == True:
    crop_start = 500
    crop_end = 1000
    img_crop = hyb_image_arr_checked[:, :, crop_start:crop_end ,crop_start:crop_end ]
    dapi_crop = register_dapi[crop_start:crop_end ,crop_start:crop_end ]
    fig, axes = plt.subplots(nrows=2, ncols=4)
    ax = axes.ravel()
    for i in range(0,hyb_image_arr_checked.shape[0]):
        print(i)
        ax[i].imshow(img_crop[i,z_ref ,:,:], vmax =np.percentile(img_crop[i,z_ref ,:,:],99.5),  cmap = 'gray')
        ax[i].axis('off')
    ax[7].imshow(dapi_crop [:,:], cmap='gray')
    ax[7].axis('off')

    hyb_image_arr_used = img_crop
else:
    hyb_image_arr_used = img_hyb_all

hyb_image_arr_used_atz = hyb_image_arr_used[:,z_start:z_end,:,:]
ref_slice = hyb_image_arr_used_atz[ref_chan,z_ref,:,:]
filter_param = [200,None] #[low cut, high cut] # we does not use this for 3D
threshold_mode = 'rel' #rel (relative value) or abs (absolute value)
dpi=200 #resolution for save image
regis_hyb_arr_2D = np.zeros((hyb_image_arr_used.shape[0], hyb_image_arr_used.shape[2], hyb_image_arr_used.shape[3]))
x_list = []
y_list = []
z_list = []
gene_list = []

#threshold_list = [0.02,0.01,0.02,0.01,0.03,0.01,0.01]
percentiile_list = [99.98,99.98,99.92,99.95,99.95,99.7,99.99]

for chan in range(0, n_channels):
    print("analysing channel number", chan)
    #data_index_list = []
    hyb_image_arr = hyb_image_arr_used_atz[chan,:,:]
    image_norm = norm_image_func(hyb_image_arr)
    filter_image = filter_func_3D(filter_param, image_norm)
    percent = percentiile_list[chan]
    threshold = np.percentile(filter_image, percent)
    #threshold = threshold_list[chan]
    print('threshold is ', threshold)
    gene_coordinates_3D = find_peak_local_max(filter_image, threshold, threshold_mode)
    sort_z_plane_from_3D_coor = gene_coordinates_3D[gene_coordinates_3D[:, 0].argsort()]  # sort coor by z
    #all_z_list = np.unique(sort_z_plane_from_3D_coor[:, 0])
    #for z in z_choose_list:
    #data_index = [i for i, x in enumerate(sort_z_plane_from_3D_coor[:, 0] == z) if x]
    #data_index_list = data_index_list + data_index
    #sort_z_plane_from_3D_coor = sort_z_plane_from_3D_coor[data_index_list]

    # To register every channel to 1 ref channel and 1 ref z before we shift all spots to that
    current_slice = hyb_image_arr[z_ref,:,:]
    register_image , shift = register_slice(ref_slice, current_slice)
    imsave(output_path + '/registered_img_channel_' + str(chan) + ".tif", register_image)
    sort_z_plane_from_3D_coor[:,1] = sort_z_plane_from_3D_coor[:,1] + shift[0]
    sort_z_plane_from_3D_coor[:,2] = sort_z_plane_from_3D_coor[:,2] + shift[1]
    # Visualize spot for 3 planes
    coor_same_z = sort_z_plane_from_3D_coor[sort_z_plane_from_3D_coor[:,0] == z_ref]
    plt.figure(figsize = (15,15))
    plt.imshow(register_image, cmap='gray')
    plt.plot(coor_same_z[:,2],coor_same_z[:,1], 'ro', markersize=1)
    plt.axis('off')
    plt.savefig(output_path + '/' + 'plot1z_channel_' + str(chan) + '.png', dpi=1000)
    plt.close()
    #plot_spot_in_3D_overlaydapi(hyb_image_arr, dapi_image_arr, sort_z_plane_from_3D_coor, output_path, dpi, alpha)
    # save spot
    #gene_name = np.array(['gene_' + str(chan)] * len(sort_z_plane_from_3D_coor)).reshape(-1,1)
    #coord_and_gene = np.hstack((sort_z_plane_from_3D_coor, gene_name))
    regis_hyb_arr_2D[chan,:,:] = register_image

    gene_name_list = ['gene_' + str(chan)] * len(sort_z_plane_from_3D_coor[:,1])
    x_list = x_list + list(sort_z_plane_from_3D_coor[:,1])
    y_list = y_list + list(sort_z_plane_from_3D_coor[:,2])
    z_list = z_list + list(sort_z_plane_from_3D_coor[:,0])
    gene_list = gene_list + gene_name_list
data = [z_list, x_list, y_list, gene_list]
df = pd.DataFrame(data)
new_df = df.T
df_tosave = new_df.set_axis([ 'z', 'y', 'x', 'gene'], axis=1, inplace=False)
df_tosave.to_csv(output_path +  "/savespot_3D_1.csv")

for i in range(0,7):
    image = hyb_image_arr_used_atz[i,z_ref,:,:]
    imsave(output_path +  "crop_image_atz=42_chan_" +str(i) +".tif", image)

# Show crop iamge after register and plus DAPI (1 z)

fig, axes = plt.subplots(nrows=2, ncols=4)
ax = axes.ravel()
for i in range(0,hyb_image_arr_checked.shape[0]):
    print(i)
    ax[i].imshow(regis_hyb_arr_2D[i,:,:], vmax = np.percentile(regis_hyb_arr_2D[i,:,:],99.5), cmap = 'gray')
    ax[i].axis('off')
ax[7].imshow(dapi_crop[:,:], cmap='gray')
ax[7].axis('off')
