import os
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog
#import pandas as pd
from tifffile import imread, imsave
import plotly.express as ex
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
#import pandas as pd
from itertools import groupby
import os
import timeit
from frequencyFilter import butter2d, butter3d

root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory(title="Please select data directory")
root.destroy()

start_time = timeit.default_timer()
image_files_list = glob.glob(data_path + '/*.tif')
output_path = os.path.join(data_path,"output_" + str(start_time))
if not os.path.exists(output_path):
    os.makedirs(output_path)

n_stack = 17
n_channel = 3
channels_to_analyse = 2 # 0 is highest wavelength (cy7), 1 is lower, 2 is lowest wavelength
analyse_mode = 'plane-by-plane' #plane-by-plane or 3D
z_plane = [6] #for plane by plane only (which z to analyse)
z_range = list(range(6,13)) #for 3D only
filter_param = [350, 900] #[low cut, high cut]
threshold_mode = 'rel' #rel or abs
threshold = 0.4 #0-1 for rel and any number for abs
dpi=200 #resolution for save image #1000 is enough for high res
color = 'm.' #'b.', 'g.', 'r.', 'c.', 'm.'


def check_channels_and_z(image_files, n_stack, n_channel, z, chan):
    tiff_img = imread(image_files)
    x_dim = np.max(tiff_img.shape)
    y_dim = np.max(tiff_img.shape)
    print("reading file from", image_files)

    assert tiff_img.ndim == 4, (
        f"tiff file:\n{image_files}\n "
        f"has {tiff_img.ndim} dimensions. Should be 4.\n"
    )
    if tiff_img.shape[0] > tiff_img.shape[1]:
        tiff_img_new = np.moveaxis(tiff_img, [0, 1, 2, 3], [2, 3, 0, 1])
        tiff_img_new = tiff_img_new.reshape(x_dim, y_dim, n_stack*n_channel)
        tiff_img_new = tiff_img_new.reshape(x_dim, y_dim, n_channel, n_stack)
        tiff_img_new = np.moveaxis(tiff_img_new, [0, 1, 2, 3], [2, 3, 0, 1])
        temp_img = tiff_img_new[:, z, ...]

    if tiff_img.shape[0] < tiff_img.shape[1]:
        tiff_img_new = tiff_img
        temp_img = tiff_img[:,z,...]

    return temp_img, tiff_img_new

def norm_image_func(image):
    """" Params: image (2 or 3D arr)
         Return: normalise image arr by min = 0 and max = 1 (2 or 3D arr)
    """
    image_max = np.max(image)
    image_min = np.min(image)
    print("minimum intensity of image =", image_min)
    print("maximum intensity of image =", image_max)
    return (image - image_min) / (image_max - image_min)

def find_peak_local_max(image_arr, thres, thres_mode):
    """ Params: image (2D or 3D arr) : image (z,x,y) or (x,y)
                threshold (float) : absolute threshold cutoff
        Return: coordinates (z,x,y) or (x,y) of peaks
    """
    if thres_mode == 'rel':
        coordinates = peak_local_max(image_arr, min_distance=1, threshold_rel=thres)
    if thres_mode == 'abs':
        coordinates = peak_local_max(image_arr, min_distance=1, threshold_abs=thres)
    return coordinates

def plot_spot_in_3D(image3D, coor_list, output_path, dpi):
    """ Params: image3D (3D arr): image (z,x,y)
                coor_list : list of coordinates that we want to plot
                coor_num : number of coordinates that we want to plot
        Return: plot image
    """
    z_list = np.unique(coor_list[:,0])
    save_path = os.path.join(output_path, 'spots at every z')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for z in z_list:
        plt.figure()
        plt.title('Spots at z = ' + str(z+1))
        coor_same_z = coor_list[coor_list[:,0] == z]
        plt.imshow(image3D[z, :], cmap=plt.cm.gray)
        plt.plot(coor_same_z[:,2], coor_same_z[:,1], 'r.')
        plt.savefig(save_path + '/' + 'spot_num' + '_3D_at_z_' + str(z) + '.jpg', dpi=dpi)

def plot_spot_in_3D_filter(image3D, filter_img, coor_list, output_path, dpi):
    """ Params: image3D (3D arr): image (z,x,y)
                coor_list : list of coordinates that we want to plot
                coor_num : number of coordinates that we want to plot
        Return: plot image
    """
    z_list = np.unique(coor_list[:, 0])
    save_path = os.path.join(output_path, 'spots at every z')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for z in z_list:
        coor_same_z = coor_list[coor_list[:, 0] == z]
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        axes[0].imshow(filter_img[z,:], cmap=plt.cm.gray)
        axes[0].axis('off')
        axes[0].set_title('Filtered Image')
        axes[1].imshow(image3D[z, :], cmap=plt.cm.gray)
        axes[1].axis('off')
        axes[1].set_title('Raw Image')
        axes[2].imshow(image3D[z, :], cmap=plt.cm.gray)
        axes[2].axis('off')
        axes[2].set_title('Spot coordinate at z' +str(z))
        axes[2].plot(coor_same_z[:,2], coor_same_z[:,1], color)
        plt.show()
        plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(
            z) + '_3D_spots.png', dpi=dpi)


def plot_z_notin2D(image3D, MIPimage, coor_list, output_path):
    """ Params: image3D (3D arr): image (z,x,y)
                MIPimage : (2D arr) : MIP image
                coor_list : list of coordinates that we want to plot

        Return: plot image
    """
    z_list = np.unique(coor_list[:, 0])
    save_path = os.path.join(output_path, 'spots only in 3D')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for z in z_list:
        plt.figure()
        plt.title('Spots at z = ' + str(z+1))
        coor_same_z = coor_list[coor_list[:,0] == z]
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(image3D[z,:], cmap=plt.cm.gray, vmin = 0)
        ax[0].axis('off')
        ax[0].set_title('Image at z' + str(z))
        ax[1].imshow(MIPimage, cmap=plt.cm.gray, vmin=0)
        ax[1].axis('off')
        ax[1].set_title('MIP')
        ax[2].imshow(image3D[z,:], cmap=plt.cm.jet)
        ax[2].axis('off')
        ax[2].plot(coor_same_z[:,2], coor_same_z[:,1], 'r.')
        #plt.show()
        plt.savefig(save_path + '/' + 'spots not in 2D at z' + str(z) + '.tif')

def filter_func_3D(filter_param, img_arr):
    freq_filter = butter3d(low_cut=filter_param[0], high_cut=filter_param[1],  # filter_path=os.path.join(data_path, "filters"),
                           order=2, zdim=img_arr.shape[0], xdim=img_arr.shape[1], ydim=img_arr.shape[2])
    filter_shifted = np.fft.fftshift(freq_filter)
    img_fourier = np.fft.fftn(img_arr)
    filtered_img = np.fft.ifftn(img_fourier * filter_shifted)
    return filtered_img.real

def filter_func(filter_param, img_arr):
    freq_filter = butter2d(low_cut=filter_param[0], high_cut=filter_param[1],  # filter_path=os.path.join(data_path, "filters"),
                           order=2, xdim=img_arr.shape[0], ydim=img_arr.shape[1])
    filter_shifted = np.fft.fftshift(freq_filter)
    img_fourier = np.fft.fftn(img_arr)
    filtered_img = np.fft.ifftn(img_fourier * filter_shifted)
    return filtered_img.real

if analyse_mode == 'plane-by-plane':
    for image_file in image_files_list:
        filename = (image_file.split("\\")[-1]).split(".")[0]
        print("image_file..", filename)
        for z_num in z_plane:
            image_arr, _ = check_channels_and_z(image_file, n_stack, n_channel, z_num, channels_to_analyse)
            image_arr = image_arr[channels_to_analyse,:]
            image_norm = norm_image_func(image_arr)
            filter_image = filter_func(filter_param, image_arr)
            gene_coordinates = find_peak_local_max(filter_image, threshold, threshold_mode)
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
            axes[0].imshow(np.asarray(filter_image).reshape(image_arr.shape[1],image_arr.shape[1]), cmap=plt.cm.gray)
            axes[0].axis('off')
            axes[0].set_title('Filtered Image')
            axes[1].imshow(image_arr,cmap=plt.cm.gray)
            axes[1].axis('off')
            axes[1].set_title('Raw Image')
            axes[2].imshow(image_arr, cmap=plt.cm.gray)
            axes[2].axis('off')
            axes[2].set_title('Spot coordinate')
            axes[2].plot(gene_coordinates[:, 1], gene_coordinates[:, 0], color)
            plt.show()
            plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots.png')

            plt.figure()
            plt.imshow(image_arr,cmap=plt.cm.gray)
            plt.axis('off')
            plt.title('Spot coordinate of genes')
            plt.plot(gene_coordinates[:, 1], gene_coordinates[:, 0], color)
            plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots_2.png', dpi=dpi)

if analyse_mode == '3D':
    for image_file in image_files_list:
        filename = (image_file.split("\\")[-1]).split(".")[0]
        print("image_file..", filename)
        _, image_arr_3D = check_channels_and_z(image_file, n_stack, n_channel, 0, channels_to_analyse)
        image_arr_3D_select = image_arr_3D[channels_to_analyse,z_range,:]
        image_norm = norm_image_func(image_arr_3D_select)
        filter_image = filter_func_3D(filter_param, image_arr_3D_select)
        gene_coordinates_3D = find_peak_local_max(filter_image, threshold, threshold_mode)
        sort_z_plane_from_3D_coor = gene_coordinates_3D[gene_coordinates_3D[:, 0].argsort()]  # sort coor by z
        plot_spot_in_3D(image_arr_3D_select, sort_z_plane_from_3D_coor, output_path, dpi)
        plot_spot_in_3D_filter(image_arr_3D_select, filter_image, sort_z_plane_from_3D_coor, output_path, dpi)

# # save spot
# df = pd.DataFrame(coordinates)
# df.to_csv(output_path + Main_name + "savespot_localmax_MIP.csv")
# df2 = pd.DataFrame(sort_z_plane_from_3D_coor)
# df2.to_csv(output_path + Main_name + "savespots_3D.csv")
# df3 = pd.DataFrame(list_3D) # list of spots that only find in 3D
# df3.to_csv(output_path + Main_name + "savespots_only in 3D.csv")
# df4 = pd.DataFrame(list_MIP) # list of spots that only find in 3D
# df4.to_csv(output_path + Main_name + "savespots_only in MIP.csv")





