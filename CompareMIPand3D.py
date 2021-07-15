from scipy import signal
from scipy import misc
from tifffile import imread, imsave
import plotly.express as ex
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby
import os
import trackpy


def norm_image_func(image):
    """" Params: image (2 or 3D arr)
         Return: normalise image arr by min = 0 and max = 1 (2 or 3D arr)
    """
    image_max = np.max(image)
    #image_min = np.min(image)
    image_min = 78 # 78 for AF594, 83 for cy5 and  91 for cy7
    image_min = 78 # 78 for AF594, 83 for cy5 and  91 for cy7
    print("min", image_min)
    print("max", image_max)
    return (image - image_min) / (image_max - image_min)

def plot3img_localmax(img1, img2, coordinates):
    """ Params: img 1 (3D arr) : image 1 of MIP
                img 2 (2D arr) : image 2 of MIP
                coordinates (:,2 arr) : corrdinates of spot finding in this peak local max
        Return: plot image
    """

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img1, cmap=plt.cm.gray)#vmin = 0, vmax=200)
    ax[0].axis('off')
    ax[0].set_title('Image 1')

    ax[1].imshow(img2, cmap=plt.cm.jet) #vmin = 0, vmax=500)
    ax[1].autoscale(False)
    ax[1].axis('off')
    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[1].set_title('peak local max')
    fig.tight_layout()
    plt.show()

def make_mips(image, output_dir):
    """ Params: image (3D arr) : image (z,x,y)
                output_dir : where image will be save as tiff
        Return: max_arrat (2D arr): Max projection intensity of image
    """
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    print("process in the image shape", image.shape)
    zmid = int(image.shape[2]/2)
    max_array = np.amax(image, axis=2)
    imsave(output_dir + 'image_MIP.png', max_array)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image[:, :, zmid], cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Image at middle z')
    ax[1].imshow(max_array, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Max Intensity Projection')
    plt.show()
    plt.savefig(output_dir+ 'plot_MIP.png')
    return max_array

def find_peak_local_max(image_arr, thres):
    """ Params: image (2D or 3D arr) : image (z,x,y) or (x,y)
                threshold (float) : absolute threshold cutoff
        Return: coordinates (z,x,y) or (x,y) of peaks
    """
    image_norm = norm_image_func(image_arr)
    coordinates = peak_local_max(image_norm, min_distance=3, threshold_rel=thres)
    return coordinates

def find_intersect(coor_arr1, coor_arr2):
    """ Params: coor_arr1 (:,2 arr) : coordinate array (x,y)
                coor_arr2 (:,2 arr) : coordinate in arr (x,y)
        Return: 1. list of coordinate that found in both array
                2. list of coordinate that only found in arr 1
                3. list of coordinate that only found in arr 2
    """
    coor_set1 = set(map(tuple, list(coor_arr1)))
    coor_set2 = set(map(tuple, list(coor_arr2)))
    intersect_set12 = coor_set1 & coor_set2
    coor_in_set1 = coor_set1 - intersect_set12
    coor_in_set2 = coor_set2 - intersect_set12
    return list(intersect_set12), list(coor_in_set1), list(coor_in_set2)

def find_xyz_of_3D(corr_3D, corr_only_tuple):
    """ Params: coor_3D (:,3) arr : 3D coordinates (z,x,y)
                corr_only_tuple (tuple): coordinates (x,y)
        Return: list of xyz that found in corr_only_tuple
    """
    corr_only_arr = np.array(corr_only_tuple)
    xyz_list = []
    for xy in corr_only_arr:
        coor_bool = xy == corr_3D[:,1:3]
        true_coor = np.prod(coor_bool, axis=1)
        z_index = np.where(true_coor)[0]
        xyz_list.append(corr_3D[z_index,:])
    return xyz_list

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

def plot_spot_in_3D(image3D, coor_list, output_path):
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
        plt.imshow(image3D[z, :], cmap=plt.cm.jet)
        plt.plot(coor_same_z[:,2], coor_same_z[:,1], 'r.')
        #plt.show()
        plt.savefig(save_path + '/' + 'spot_num' + '_3D_at_z_' + str(z) + '.tif')



def possible_of_xy_nearby(coor_only_in3D, coor_intersect):
    """ Params: coor_only_in3D (list) : (x,y) of coor that only find in 3D not MIP
                coor_MIP (array) : (x,y) of coor that find in MIP
        Return: coordinates in MIP that nearby coor in 3D
    """
    coor_only_in3D = np.array(coor_only_in3D)
    xy_nearby_list = []
    distance_arr_x = np.zeros((coor_intersect.shape[0],coor_only_in3D.shape[1]))
    distance_arr_y = np.zeros((coor_intersect.shape[0],coor_only_in3D.shape[1]))
    print(distance_arr_x.shape)
    sort_coor_MIP_by_x  = coor_intersect[coor_intersect[:, 0].argsort()]
    sort_coor_3D_by_x = coor_only_in3D[coor_only_in3D[:, 0].argsort()]
    for i in range(len(sort_coor_3D_by_x)):
        xy_pair_3D = sort_coor_3D_by_x[i]
        print(xy_pair_3D)
        for j in range(len(sort_coor_MIP_by_x)):
            xy_pair_MIP = sort_coor_MIP_by_x[j]
            distance_xy = xy_pair_3D-xy_pair_MIP
            distance_arr_x[j,i] = distance_xy[0]
            distance_arr_y[j,i] = distance_xy[1]

    for i in range(distance_arr_x.shape[1]):
        for x_dis,y_dis in zip(distance_arr_x[:,i], distance_arr_y[:,i]):
            if x_dis <= abs(3) and y_dis <= abs(3):


                #xy_nearby_list.append(sort_coor_MIP_by_x[index])
                print(xy_nearby_list)

    return xy_nearby_list

def xy_nearby(coor_list_1, coor_list_2, threshold):
    check_nearby_array = np.zeros((len(coor_list_1), len(coor_list_2)))
    for i in range(len(coor_list_1)):
        for j in range(len(coor_list_2)):
            dist_x = np.abs(coor_list_1[i][0] - coor_list_2[j][0])
            dist_y = np.abs(coor_list_1[i][1] - coor_list_2[j][1])
            check_nearby_array[i, j] = dist_x <= threshold and dist_y <= threshold
    list_index_1, list_index_2 = np.where(check_nearby_array)
    return list_index_1, list_index_2


def stack_images(image3D, coor_3D, output_path):
    """ Params: image3D : (z,x,y) array of image
                coor_3D : (x,y) coordinates to plot on image
        Return: tiff file of image array (z,x,y)
    """
    for i in range(len(coor_3D)):
        x = coor_3D[i][0]
        y = coor_3D[i][1]

        image_crop = image3D[:,x-10:x+10,y-10:y+10]
        imsave(output_path + "spot" + str(i) + "_crop.tiff", image_crop)



def find_spot_in_each2D(image3D, thres):
    """ Params: image (3D arr) : (z,x,y)
                thres (float) : absolute threshold for peak local max to find spots
        Returns : sum of spot in each 2D planes and can be save
     """
    coor_2D_list = []
    sum_spots = 0
    for i in range(1, image3D.shape[0]):
        print(i)
        image_norm = norm_image_func(image[i, :])
        coordinates_m = peak_local_max(image_norm, min_distance=3, threshold_rel=thres)
        coor_2D_list.append(list(coordinates_m))
        sum_spots += coordinates_m.shape[0]
    print("Sum of spots in 2D", sum_spots)
    #df4 = pd.DataFrame(coor_2D_list)
    #df4.to_csv(output_path + Main_name + "savespots_each_2D.csv")


thres = 0.23
image_path = "C:/Users/Nette/Desktop/3D_gaussian/Michelle_Test/20210511_Hyb1_F0_z37/AF594-40X_Hyb_785_iRFP_CF40_Sona1_637_Cy5_CF40_Sona1_561_RFP_CF40_Sona1_0_F0.tif"
output_path = "C:/Users/Nette/Desktop/3D_gaussian/Michelle_Test/20210511_Hyb1_F0_z37/AF594_2/"
Main_name = "AF594_thres" + str(thres)
image = imread(image_path)
imageMIP = make_mips(image, output_path) # auto save in output path
coordinates = find_peak_local_max(imageMIP, thres)
plot3img_localmax(imageMIP, imageMIP, coordinates) # save plot
plt.savefig(output_path+ 'Peak_local_max_MIP_plot.png')
coordinates_3D = find_peak_local_max(image, thres)
sort_z_plane_from_3D_coor = coordinates_3D[coordinates_3D[:, 0].argsort()] # sort coor by z
coordinates_3D_to_2D = sort_z_plane_from_3D_coor[:, 1:3] # not include z to compare with MIP

# find the same spots between MIP and 3D
intersect_list, list_MIP, list_3D = find_intersect(coordinates,coordinates_3D_to_2D) #compare coordinates from MIP and 3D

# find spots that only appear in 3D and plot them in specific z planes
fullxyz = find_xyz_of_3D(coordinates_3D, list_3D) #find (z,x,y) of coordinates (spots) that on;y found in 3D
#coor_num = [0,1,2,3,4] # number of spots to show in plot
fullxyz_arr = (np.array(fullxyz)).reshape(-1,3)
plot_z_notin2D(image, imageMIP, fullxyz_arr, output_path) # plot spots in specifics z

plot_spot_in_3D(image, sort_z_plane_from_3D_coor, output_path) #plot all spots in 3D

list_1_ind, list_2_ind = xy_nearby(list_3D, list(map(tuple, list(coordinates))), 3)
#stack_images(image, list_3D, output_path)
# save spot
df = pd.DataFrame(coordinates)
df.to_csv(output_path + Main_name + "savespot_localmax_MIP.csv")
df2 = pd.DataFrame(sort_z_plane_from_3D_coor)
df2.to_csv(output_path + Main_name + "savespots_3D.csv")
df3 = pd.DataFrame(list_3D) # list of spots that only find in 3D
df3.to_csv(output_path + Main_name + "savespots_only in 3D.csv")
df4 = pd.DataFrame(list_MIP) # list of spots that only find in 3D
df4.to_csv(output_path + Main_name + "savespots_only in MIP.csv")


# def make_mips_Norm_each2D(image,output_dir_image):
#
#     print(image.shape)
#
#     image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
#     zmid = image.shape[2]/2
#     #image = image[0:255, 0:255, :]
#     print(image.shape)
#     new_image_norm = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
#     for i in range((image.shape[2])):
#         image_norm = norm_image_func(image[:,i])
#         new_image_norm[:,i] = image_norm
#
#     max_array = np.amax(new_image_norm, axis=2)
#     imsave(output_dir_image,max_array)
#
#     fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
#     ax = axes.ravel()
#     ax[0].imshow(image[:,:,32], cmap=plt.cm.gray,vmin = 0, vmax = 10000)
#     ax[0].axis('off')
#     ax[0].set_title('Image at middle z')
#
#     ax[1].imshow(max_array, cmap=plt.cm.gray,vmin = 0, vmax = 10000)
#     ax[1].axis('off')
#     ax[1].set_title('Max Intensity Projection')
#     plt.show()
#     plt.savefig("C:/Users/Nette/Desktop/3D_gaussian/MIPvs3Ddecoding/Normeach2D_Cy5_crop_area2_Max_intensity_projection.png")
#     return max_array, new_image_norm


def Test_peak_local_max_with_fake_arr(image):
    image3 = imread("C:/Users/Nette/Desktop/3D_gaussian/MIPvs3Ddecoding/Confocal/slice0023_Cy5_allz_crop2.tif")
    #dim1 = image3.shape[0]
    #dim2 = image3.shape[1]
    shift_pixel = 1
    image2 = np.pad(image3, (1,0), 'constant', constant_values=0)
    image2 = image2[1:,:]
    #image2=image3
    dim1 = image2.shape[0]
    dim2 = image2.shape[1]
    zeros_arr = np.zeros((dim1,dim2,10))
    image_stack2 = np.dstack((image2,image2))
    image_stack_half = np.dstack((image_stack2,zeros_arr))
    image_stack2_1 = np.pad(image3, (0,1), 'constant', constant_values=(0,0))
    image_stack2_1 = image_stack2_1[0:36,:]
    #image_stack2_1 = image_stack2
    image_stack_full = np.dstack((image_stack_half, image_stack2_1))
    image_stack_full = np.dstack((image_stack_full , image_stack2_1))
    #image_stack_full  = np.moveaxis(image_stack_full , [0, 1, 2], [1, 0, 2])
    image_MIP = make_mips(image_stack_full, "C:/Users/Nette/Desktop/3D_gaussian/MIPvs3Ddecoding/Confocal/test_slice23_1pixel")
    Main_name2 = "Test"
    coordinates = find_peak_local_max(image_MIP, Main_name2, 0.9)
    coordinates_3D = find_peak_local_max(image_stack_full , 0.9)
    # coordinates_3D_to_2D = coordinates_3D[:, 1:3]
    # intersect_list, list_MIP, list_3D = find_intersect(coordinates,coordinates_3D_to_2D)
    # fullxyz = find_z_of_intersect(coordinates_3D, list_3D)
    # coor_num = [1]
    # plot_z_notin2D(image2, image_MIP, fullxyz, coor_num)

ascent = image[30,:,:]
from scipy.ndimage import gaussian_laplace, gaussian_filter
fig = plt.figure()
plt.gray()
result = gaussian_filter(ascent, sigma=0.3)
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()


from tifffile import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

dapi_img_2D = imread("C:/Users/Nette/Desktop/3D_gaussian/Michelle_Test/20210511_Hyb1_F0_z37/DAPI_40X_405_DAPI_CF40_Sona 1_1_F0.tif")
image = dapi_img_2D[30,:,:]
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Image')
ax[1].imshow(thresh, cmap=plt.cm.jet)
ax[1].set_title('Threshold')

# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(thresh)
local_max = peak_local_max(distance_map, indices=False, min_distance=10, labels=thresh)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(distance_map, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()



