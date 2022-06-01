from utils import register_translation
from tifffile import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import scipy

def threeD_translation(img,
                    shifts):

    """
    translate the 3D image

    """
    initial_shape = img.shape
    for di, i in enumerate(shifts):
        if i > 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((z, img), axis=0)
            img = img[:initial_shape[di],...]

        elif i < 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((img,z), axis=0)
            img = img[-initial_shape[di]:,...]
        else:
            continue
        img = np.moveaxis(img, 0, di)

    assert img.shape == initial_shape
    return img

def merge_image(image1, image2):
    # temporary RGB array to compare the reference and offset images
    rgb_temp = np.zeros(
        (image1.shape[0], image1.shape[1], 3),
        dtype=np.float32,
    )
    ref_temp = image1
    ref_min = np.min(ref_temp)
    ref_max = np.percentile(ref_temp, 99.99)
    ref_norm = (ref_temp - ref_min) / (ref_max - ref_min)

    reg_temp = image2
    reg_min = np.min(image2)
    reg_max = np.percentile(reg_temp, 99.99)
    reg_norm = (reg_temp - reg_min) / (reg_max - reg_min)

    rgb_temp[:, :, 1] = (ref_temp - ref_min) / (ref_max - ref_min)
    rgb_temp[:, :, 0] = (reg_temp - reg_min) / (reg_max - reg_min)
    #rgb_temp[:, :, 2] = ref_norm + (reg_temp - reg_min) / (reg_max - reg_min)

    return rgb_temp

mainpath = 'Y:/Jeeranan_analysis_KMS/N2743_P250/'
dapi_img = imread(mainpath + 'prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F07.tif')
hyb_img = imread(mainpath + 'test_785_Sona1_CF40_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F07.tif')

def _3D_translation(img,
                    shifts):

    """
    translate the 3D image

    """
    initial_shape = img.shape
    for di, i in enumerate(shifts):
        if i > 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((z, img), axis=0)
            img = img[:initial_shape[di],...]

        elif i < 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((img,z), axis=0)
            img = img[-initial_shape[di]:,...]
        else:
            continue
        img = np.moveaxis(img, 0, di)

    assert img.shape == initial_shape
    return img

crop_size = 200
x_cen =   953 #953
y_cen =   536 #536
dapi_image  = dapi_img[:,2,:,:]
pre_image = dapi_img[:,1,:,:]
image = hyb_img[:,2,:,:]


dapi_image_crop = dapi_image[:,x_cen-crop_size:x_cen+crop_size,y_cen-crop_size:y_cen+crop_size]
image_crop = image[:,x_cen-crop_size:x_cen+crop_size,y_cen-crop_size:y_cen+crop_size]
preimage_crop = pre_image[:,x_cen-crop_size:x_cen+crop_size,y_cen-crop_size:y_cen+crop_size]


(shifts_3d,
 fine_error,
 pixel_error) = register_translation(dapi_image_crop,
                                     preimage_crop,
                                     space="real")



(shifts_3d2,
 fine_error,
 pixel_error) = register_translation(dapi_image_crop,
                                     image_crop,
                                     space="real")


regis_image = threeD_translation(preimage_crop, shifts_3d)


#shifts_3d2 = [-6,0,0]
regis_image2 = threeD_translation(image_crop, shifts_3d2)


four_dim_image = np.zeros((75,400,400,3))
regis_four_dim_image  = np.zeros((75,400,400,3))
regis_four_dim_image_2  = np.zeros((75,400,400,3))
for i in range(75):
    #combine_img = merge_image(dapi_image[i,:,:], image[i,:,:])
    combine_img_reg =  merge_image(regis_image[i,:,:], regis_image2[i,:,:])
    #four_dim_image[i,:,:,:] = combine_img
    regis_four_dim_image[i,:,:,:] = combine_img_reg

imsave(mainpath + 'crop_regis_hyb_andprehybtodapi_hyb0_fov7_cy3.tiff', regis_four_dim_image)



#imsave(mainpath + 'image_pre_hyb.tiff', four_dim_image)
# fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
# ax= axes.ravel()
# ax[0].imshow(dapi_image[30,:,:])
# ax[1].imshow(image[30,:,:])correlate2d
# ax[2].imshow(combine_img)
# ax[3].imshow(combine_img_reg)
#

# x_shift = np.zeros((75,75))
# y_shift = np.zeros((75,75))
# phase_diff = np.zeros((75,75))
# for i in range(75):
#     dapi = dapi_image[i,:,:]
#     for j in range(75):
#         hyb = image[j,:,:]
#         shifts,_, phase = register_translation(dapi,hyb,space="real")
#         x_shift[i,j] = shifts[0]
#         y_shift[i,j] = shifts[1]
#         phase_diff[i,j] = phase

#
# min_xshift_ind = np.where(x_shift == np.min(abs(x_shift)))
# diffz_xshift = min_xshift_ind[0] - min_xshift_ind[1]
# min_yshift_ind = np.where(y_shift == np.min(abs(y_shift)))
# diffz_yshift = min_yshift_ind[0] - min_yshift_ind[1]
#
#
# fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
# ax = axes.ravel()
# ax[0].imshow(x_shift[0:75,0:75], cmap = 'rainbow')
# for i in range(75):
#     for j in range(75):
#         if x_shift[i,j] == 0:
#             ax[0].annotate('0',
#             xy=(j, i), xycoords='data',
#             xytext=(0, 0), textcoords='offset pixels',)
# ax[1].imshow(y_shift[0:75,0:75], cmap = 'rainbow')
# for i in range(75):
#     for j in range(75):
#         if y_shift[i,j] == 0:
#             ax[1].annotate('0',
#                          xy=(j, i), xycoords='data',
#                          xytext=(0, 0), textcoords='offset pixels',)
# plt.gca().invert_yaxis()
# plt.colorbar()
#
