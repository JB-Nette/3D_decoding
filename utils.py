"""
Mod of Scikit-Image's register_translation function

Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
"""
from scipy import ndimage
import numpy as np
from frequencyFilter import butter2d, butter3d
from skimage.feature import peak_local_max
from tifffile import imread, imsave


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                  * np.fft.fftfreq(n_items, upsample_factor))
        kernel = np.exp(-im2pi * kernel)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / \
            (src_amp * target_amp)
    return np.sqrt(np.abs(error))


def register_translation(src_image: np.ndarray,
                         target_image: np.ndarray,
                         upsample_factor: int = 1,
                         space: str = "real",
                         return_error: bool = True,
                         ):
    """
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    return_error : bool, optional
        Returns error and phase difference if on,
        otherwise only shifts are returned
    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fftn(src_image)
        target_freq = np.fft.fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # calculate the pixel-resolution maximum cross-correlation value
    CCmax_pixel = cross_correlation[maxima]

    if upsample_factor == 1:
        if return_error:
            src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
            target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
            CCmax = CCmax_pixel # this is height of cc peak

    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)

        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)

        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization

        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        CCmax = cross_correlation[maxima]

        maxima = np.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor

        if return_error:
            src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                     1, upsample_factor)[0, 0]
            src_amp /= normalization
            target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                        1, upsample_factor)[0, 0]
            target_amp /= normalization

    # For singleton dimensions, the shift calculated has no effect.
    # Set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        # return (shifts,
        #         _compute_error(CCmax, src_amp, target_amp),
        #         _compute_phasediff(CCmax)
        #         )

        return (shifts,
                np.abs(CCmax),
                np.abs(CCmax_pixel)
                )
    else:
        return shifts

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

def check_channels_and_output_multiplez(tiff_img, n_stack, n_channel):

    x_dim = np.max(tiff_img.shape)
    y_dim = np.max(tiff_img.shape)


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
        coordinates = peak_local_max(image_arr, min_distance=5, threshold_rel=thres)
    if thres_mode == 'abs':
        coordinates = peak_local_max(image_arr, min_distance=5, threshold_abs=thres)
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

def filter_func_3D(filter_param, img_arr):
    freq_filter = butter3d(low_cut=filter_param[0], high_cut=filter_param[1],  # filter_path=os.path.join(data_path, "filters"),
                           order=2, zdim=img_arr.shape[0], xdim=img_arr.shape[1], ydim=img_arr.shape[2])
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

