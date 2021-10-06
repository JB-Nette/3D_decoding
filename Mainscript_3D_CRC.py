import os
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
from tifffile import imread, imsave
import numpy as np
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import os
import timeit
from frequencyFilter import butter2d
from utils import register_translation
from scipy import ndimage

root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory(title="Please select data directory")
root.destroy()

start_time = timeit.default_timer()
image_files_list = glob.glob(data_path + '/*.tif')
output_path = os.path.join(data_path,"output_" + str(start_time))
if not os.path.exists(output_path):
    os.makedirs(output_path)

hyb_image_file = "C:/Users/Nette/Desktop/3D_gaussian/hyb_00_CRC/hyb00_series12.tif"
dapi_image_file = "C:/Users/Nette/Desktop/3D_gaussian/hyb_00_CRC/DAPI_series12.tif"
n_stack = 9 # total of z stack in hyb image
n_channel = 4 # total of z stack in hyb image
dapi_channel_number = 1 # the number of channel that dapi located is 1
channels_to_analyse = 2 # refer to gene, 0 is highest wavelength, 1 is lower, 2 is lowest wavelength
alpha = 0.5 # 0 is transparent, 1 is  to show the overlay image of dapi and hyb
z_choose_list = list(range(3,6)) #what z do you want to show for 3D only # minimum is 2 z
filter_param = [100,None] #[low cut, high cut] # we does not use this for 3D
threshold_mode = 'rel' #rel (relative value) or abs (absolute value)
threshold = 0.1 #0-1 for rel and any number for abs
dpi=200 #resolution for save image

methods = '3D' # planebyplane or 3D


def check_channels_and_z(image_files, n_stack, n_channel, z):
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

def check_channels_and_output_multiplez(image_files, n_stack, n_channel):
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
        temp_img = tiff_img_new[:, :, ...]

    if tiff_img.shape[0] < tiff_img.shape[1]:
        tiff_img_new = tiff_img
        temp_img = tiff_img[:,:,...]

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

def plot_spot_in_3D_overlaydapi(image3D, image3D_dapi, coor_list, output_path, dpi, alpha =0.4):
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
        register_dapi, _ = register_slice(image3D[z, :],image3D_dapi[z, :])
        print(register_dapi.shape)
        coor_same_z = coor_list[coor_list[:, 0] == z]
        fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)
        axes = axes.ravel()
        axes[0].imshow(norm_image_func(image3D[z, :]),vmax=0.4,cmap ='gray')
        axes[0].axis('off')
        axes[0].set_title('Hyb image at z ='+ str(z))
        axes[1].imshow(norm_image_func(image3D_dapi[z, :]), cmap='Purples', vmax=np.percentile(norm_image_func(image3D_dapi[z, :]),99.95))
        axes[1].set_title('Dapi image at z =' + str(z))
        axes[1].axis('off')
        axes[2].imshow(norm_image_func(image3D[z, :]),vmax=0.4)
        axes[2].imshow(norm_image_func(image3D_dapi[z,:]), cmap='Purples',alpha=alpha)
        axes[2].set_title('Hyb with Dapi image at z ='+ str(z))
        axes[2].axis('off')
        axes[3].imshow(norm_image_func(image3D[z, :]), vmax=0.4,cmap ='gray')
        axes[3].set_title('Hyb with spot coordinates at z ='+ str(z))
        axes[3].plot(coor_same_z[:, 2], coor_same_z[:, 1], 'r.')
        axes[3].axis('off')
        axes[4].imshow(norm_image_func(image3D[z, :]), vmax=0.4)
        axes[4].imshow(norm_image_func(image3D_dapi[z,:]), cmap='Purples', alpha=alpha)
        axes[4].set_title('Hyb with Dapi image with \n spot coordinates at z ='+ str(z))
        axes[4].plot(coor_same_z[:, 2], coor_same_z[:, 1], 'r.')
        axes[4].axis('off')
        axes[5].imshow(norm_image_func(image3D[z, :]), vmax=0.4)
        axes[5].imshow(norm_image_func(register_dapi), cmap='Purples', alpha=alpha)
        axes[5].set_title('Hyb with Registered Dapi image with \n spot coordinates at z =' + str(z))
        axes[5].plot(coor_same_z[:, 2], coor_same_z[:, 1], 'r.')
        axes[5].axis('off')
        plt.show()
        plt.savefig(save_path + '/' + 'spot_num' + '_3D_at_z_' + str(z) + '.jpg', dpi=dpi)

    fig, axes = plt.subplots(3, len(z_list), figsize=(10,10), sharex=True, sharey=True)
    print(z_list)
    for i, z in enumerate(z_list):
        register_dapi, _ = register_slice(image3D[z, :], image3D_dapi[z, :])
        print(image3D_dapi.shape)
        coor_same_z = coor_list[coor_list[:, 0] == z]
        axes[0,i].imshow(norm_image_func(register_dapi), cmap='Purples')
        axes[0, i].axis('off')
        axes[0, i].set_title('Registered DAPI at z=' + str(z))
        axes[1, i].imshow(norm_image_func(image3D[z, :]), vmax=0.3, cmap ='gray')
        axes[1, i].plot(coor_same_z[:, 2], coor_same_z[:, 1], 'r.')
        axes[1, i].set_title('Hyb at z=' + str(z))
        axes[1, i].axis('off')
        axes[2,i].imshow(norm_image_func(image3D[z, :]), vmax=0.3)
        axes[2,i].imshow(norm_image_func(register_dapi), cmap='Purples', alpha=alpha)
        axes[2,i].plot(coor_same_z[:, 2], coor_same_z[:, 1], 'r.')
        axes[2,i].set_title('At z=' + str(z))
        axes[2,i].axis('off')

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

def filter_func(filter_param, img_arr):
    freq_filter = butter2d(low_cut=filter_param[0], high_cut=filter_param[1],  # filter_path=os.path.join(data_path, "filters"),
                           order=2, xdim=img_arr.shape[1], ydim=img_arr.shape[1])
    filter_shifted = np.fft.fftshift(freq_filter)
    img_fourier = np.fft.fftn(img_arr)
    filtered_img = np.fft.ifftn(img_fourier * filter_shifted)
    return filtered_img.real

def register_slice(ref_slice, current_slice, shifts=None):
    ref_slice_fourier = np.fft.fftn(ref_slice)
    current_slice_fourier = np.fft.fftn(current_slice)
    (shifts, fine_error,  pixel_error) = register_translation(ref_slice_fourier,
                                                 current_slice_fourier,
                                                 upsample_factor=50,
                                                 space="fourier")
    print("shifts = ", shifts)

    registered_slice = np.fft.ifftn(ndimage.fourier_shift(current_slice_fourier, shifts))
    return registered_slice.real, shifts


dapi_image_arr = imread(dapi_image_file)[:, dapi_channel_number ,:,:] #dapi images seem correct
hyb_image_arr, _ = check_channels_and_output_multiplez(hyb_image_file, n_stack, n_channel)
hyb_image_arr = hyb_image_arr[channels_to_analyse,:,:]
if methods == 'planebyplane':
    for z_num in range(n_stack):
        dapi_image_arr_zth = dapi_image_arr[z_num, :, :]
        hyb_image_arr_zth = hyb_image_arr[z_num, :, :]
        register_hyb_img, _ = register_slice(dapi_image_arr_zth,hyb_image_arr_zth)
        image_norm = norm_image_func(register_hyb_img)
        filter_image = filter_func(filter_param, image_norm)
        gene_coordinates = find_peak_local_max(filter_image, threshold, threshold_mode)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        axes = axes.ravel()
        axes[0].imshow(image_norm)
        axes[0].axis('off')
        axes[0].set_title('Hyb image at z ='+ str(z_num))
        axes[1].imshow(image_norm,vmax=0.4)
        axes[1].imshow(norm_image_func(dapi_image_arr_zth), cmap='Purples',alpha=alpha)
        axes[1].set_title('Hyb with Dapi image at z ='+ str(z_num))
        axes[1].axis('off')
        axes[2].imshow(image_norm, vmax=0.4)
        axes[2].set_title('Hyb with spot coordinates at z ='+ str(z_num))
        axes[2].plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
        axes[2].axis('off')
        axes[3].imshow(image_norm, vmax=0.4)
        axes[3].imshow(norm_image_func(dapi_image_arr_zth), cmap='Purples', alpha=alpha)
        axes[3].set_title('Hyb with Dapi image with spot coordinates at z ='+ str(z_num))
        axes[3].plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
        axes[3].axis('off')
        plt.show()


data_index_list = []
if methods == '3D':
    #register_hyb_img, _ = register_slice(dapi_image_arr_zth, hyb_image_arr_zth)
    image_norm = norm_image_func(hyb_image_arr)
    gene_coordinates_3D = find_peak_local_max(image_norm, threshold, threshold_mode)
    sort_z_plane_from_3D_coor = gene_coordinates_3D[gene_coordinates_3D[:, 0].argsort()]  # sort coor by z
    all_z_list = np.unique(sort_z_plane_from_3D_coor[:, 0])
    for z in z_choose_list:
        if z in all_z_list:
            print(z)
            data_index = [i for i, x in enumerate(sort_z_plane_from_3D_coor[:, 0] == z) if x]
            data_index_list = data_index_list + data_index
            print(len(data_index_list))
    sort_z_plane_from_3D_coor = sort_z_plane_from_3D_coor[data_index_list]
    plot_spot_in_3D_overlaydapi(hyb_image_arr, dapi_image_arr, sort_z_plane_from_3D_coor, output_path, dpi, alpha)

    # save spot
    df = pd.DataFrame(sort_z_plane_from_3D_coor )
    df = df.set_axis([ 'z', 'y', 'x'], axis=1, inplace=False)
    df.to_csv(output_path + '/threshold ' + str(threshold) + "savespot_3D.csv")




# if analyse_mode == 'plane-by-plane':
#     for image_file in image_files_list:
#         filename = (image_file.split("\\")[-1]).split(".")[0]
#         print("image_file..", filename)
#         for z_num in z_plane:
#             image_arr, _ = check_channels_and_z(image_file, n_stack, n_channel, z_num, channels_to_analyse)
#             image_arr = image_arr[channels_to_analyse,:]
#             image_norm = norm_image_func(image_arr)
#             filter_image = filter_func(filter_param, image_arr)
#             gene_coordinates = find_peak_local_max(filter_image, threshold, threshold_mode)
#             fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
#             axes[0].imshow(np.asarray(filter_image).reshape(image_arr.shape[1],image_arr.shape[1]), cmap=plt.cm.gray)
#             axes[0].axis('off')
#             axes[0].set_title('Filtered Image')
#             axes[1].imshow(image_arr,cmap=plt.cm.gray)
#             axes[1].axis('off')
#             axes[1].set_title('Raw Image')
#             axes[2].imshow(image_arr, cmap=plt.cm.jet)
#             axes[2].axis('off')
#             axes[2].set_title('Spot coordinate')
#             axes[2].plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
#             plt.show()
#             plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots.png')
#
#             plt.figure()
#             plt.imshow(image_arr,cmap=plt.cm.gray)
#             plt.axis('off')
#             plt.title('Spot coordinate of genes')
#             plt.plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
#             plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots_2.png', dpi=dpi)
#
# if analyse_mode == '3D':
#     for image_file in image_files_list:
#         filename = (image_file.split("\\")[-1]).split(".")[0]
#         print("image_file..", filename)
#         _, image_arr_3D = check_channels_and_z(image_file, n_stack, n_channel, 0, channels_to_analyse)
#         image_arr_3D_select = image_arr_3D[channels_to_analyse,3:15,:]
#         image_norm = norm_image_func(image_arr_3D_select)
#         #filter_image = filter_func(filter_param, image_arr)
#         gene_coordinates_3D = find_peak_local_max(image_norm, threshold, threshold_mode)
#         sort_z_plane_from_3D_coor = gene_coordinates_3D[gene_coordinates_3D[:, 0].argsort()]  # sort coor by z
#         plot_spot_in_3D(image_arr_3D_select, sort_z_plane_from_3D_coor, output_path, dpi)
        # fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        # axes[0].imshow(np.asarray(filter_image).reshape(image_arr.shape[1],image_arr.shape[1]), cmap=plt.cm.gray)
        # axes[0].axis('off')
        # axes[0].set_title('Filtered Image')
        # axes[1].imshow(image_arr,cmap=plt.cm.gray)
        # axes[1].axis('off')
        # axes[1].set_title('Raw Image')
        # axes[2].imshow(image_arr, cmap=plt.cm.jet)
        # axes[2].axis('off')
        # axes[2].set_title('Spot coordinate')
        # axes[2].plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
        # plt.show()
        # plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots.png')
        #
        # plt.figure()
        # plt.imshow(image_arr,cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.title('Spot coordinate of genes')
        # plt.plot(gene_coordinates[:, 1], gene_coordinates[:, 0], 'r.')
        # plt.savefig(output_path + '/' + filename + "channels_" + str(channels_to_analyse) + "at_z_" + str(z_num) + '_plane-by-plne_spots_2.png')
# #stack_images(image, list_3D, output_path)






