import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    m, n = kernel.shape
    new_image = np.empty(img.shape)

    if len(img.shape) == 2:
        # If the image is a gray-scale image
        image_height, image_width = img.shape
        pseudo_image = np.zeros((image_height + m - 1, image_width + n - 1))
        pseudo_image[(m - 1) / 2:image_height + (m - 1) / 2, (n - 1) / 2:image_width + (n - 1) / 2] = img
        for i in range(image_width):
            for j in range(image_height):
                temp = kernel * pseudo_image[j:j + m, i:i + n]
                new_image[j, i] = temp.sum()

    elif len(img.shape) == 3:
        # If the image is an rgb image
        image_height, image_width, color_axis = img.shape
        pseudo_image = np.zeros((image_height + m - 1, image_width + n - 1, color_axis))
        pseudo_image[(m - 1) / 2:image_height + (m - 1) / 2, (n - 1) / 2:image_width + (n - 1) / 2] = img
        for i in range(image_width):
            for j in range(image_height):
                for k in range(color_axis):
                    temp = kernel * pseudo_image[:, :, k][j:j + m, i:i + n]
                    new_image[:, :, k][j, i] = temp.sum()

    return (new_image)

    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    flipped_image = np.flip(kernel, (0, 1))
    return cross_correlation_2d(img, flipped_image)

    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    pseudo_image = np.zeros((height, width))

    c_x = int(height/2)
    c_y = int(width/2)
    const = 1/(2*np.pi*(sigma**2))
    exp_den = 2*(sigma**2)

    for x in range(height):
        for y in range(width):
            exp_num = ((x - c_x)**2) + ((y - c_y)**2)
            exponent = (-1)*exp_num/float(exp_den)
            pseudo_image[x, y] = const * np.exp(exponent)

    gaussian_blur = pseudo_image/pseudo_image.sum()
    return gaussian_blur

    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    low_pass_image = convolve_2d(img, kernel)
    return low_pass_image

    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    high_pass_image = img - low_pass(img, sigma, size)
    return high_pass_image
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

