"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB_SHAPE = 3
def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207054354


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
   #Gray scale
    if representation == LOAD_GRAY_SCALE:
        img1 = cv2.imread(filename)
       ## img1 = img1.astype(np.float32)
        return cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #RGB
    elif representation == LOAD_RGB:
        img = cv2.imread(filename)
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation == LOAD_GRAY_SCALE:
        img = cv2.imread(filename , cv2.IMREAD_GRAYSCALE)
        plt.imshow(img,cmap='gray')

    elif representation == LOAD_RGB:
        img1 = cv2.imread(filename)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        plt.imshow(img1)

    plt.show()
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :rtype: object
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape) / 255


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transMatrix = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                          [0.59590059, -0.27455667, -0.32134392],
                                          [0.21153661, -0.52273617, 0.31119955]])).transpose()
    shape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), transMatrix).reshape(shape)
    pass


def hist_eq(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function will do histogram equalization on a given 1D np.array
    meaning will balance the colors in the image.
    For more details:
    https://en.wikipedia.org/wiki/Histogram_equalization
    **Original function was taken from open.cv**
    :param img: a 1D np.array that represent the image
    :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    # Flattning the image and converting it into a histogram
    histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])
    # Calculating the cumsum of the histogram
    cdf = histOrig.cumsum()
    # Places where cdf = 0 is ignored and the rest is stored
    # in cdf_m
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalizing the cdf
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Filling it back with zeros
    cdf = np.ma.filled(cdf_m, 0)

    # Creating the new image based on the new cdf
    imgEq = cdf[img.astype('uint8')]
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    return imgEq, histOrig, histEq
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        The function will fist check if the image is RGB or gray scale
        If the image is gray scale will equalizes
        If RGB will first convert to YIQ then equalizes the Y level
        :param imgOrig: Original Histogram
        :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    if len(imgOrig.shape) == 2:
        img = imgOrig * 255
        imgEq, histOrig, histEq = hist_eq(img)

    else:
        img = transformRGB2YIQ(imgOrig)
        #print(img[0].shape)
        img[:, :, 0] = img[:, :, 0] * 255
        im_copy = (img)[:,:,0]
        imgEq, histOrig, histEq = hist_eq(im_copy)
        imgEq =imgEq / 255
        imgEq = transformYIQ2RGB(np.stack([imgEq, img[:,:, 1], img[:, :, 2]], axis=2))
    return imgEq, histOrig, histEq


def quantizeImage(im_orig: np.ndarray, n_quant: int, n_iter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    error = np.array(list())
    temp = []
    rgb_flag = False
    im_copy = im_orig.copy().astype(np.float64) \
        if im_orig.dtype != np.float64 else im_orig.copy()
    if len(im_copy.shape) == RGB_SHAPE:
        # if the image is rgb
        temp = transformRGB2YIQ(im_copy)
        im_copy = (temp)[:, :, 0]
        rgb_flag = True
    im_copy_255 = (im_copy * 255).astype(np.uint8)
    hist_orig, bins = np.histogram(im_copy_255, bins=range(0, 257))
    # calculating the first segments
    new_z = np.rint(np.quantile(im_copy_255, np.linspace(0, 1, n_quant + 1))).astype(np.uint8)
    new_z[0] = 0
    new_z[-1] = 255
    z = new_z.copy()
    quantums = np.zeros(n_quant, dtype=np.float64)
    for i in range(n_iter):
        quantums = calculate_quantum_values(hist_orig, new_z, quantums)
        new_z = calculate_segments(new_z, quantums)
        if np.array_equal(z, new_z):
            break
        error = np.append(error, calculate_error(new_z, hist_orig, quantums, n_quant))
        z = new_z.copy()
    digitized_map = np.digitize(range(0, 256), new_z, True) - 1
    digitized_map[0] = 0
    # The look up table
    def look_up_table(digitized_pixel):
        """
        This function is the lookup table needed to convert all the pixels
        at the photo to their new bins.
        :param n_quant: is the number of intensities the output
        im_quant image should have.
        :param quantums: The current quantums array
        :return: The new value of the pixel
        """
        return quantums[digitized_map[digitized_pixel]]

    apply_look_up_table = np.vectorize(look_up_table)

    im_quant = apply_look_up_table(im_copy_255) / 255

    if rgb_flag:
        im_quant = transformYIQ2RGB(

            np.stack([im_quant, temp[:, :, 1], temp[:, :, 2]], axis=2))

    return [im_quant, np.array(error)]
    pass
def calculate_segments(z, quantums):
    """
    This function calculates the new bins according to the current quantums.
    :param n_quant: is the number of intensities the output
        im_quant image should have.
    :param quantums: The quantums array.
    :param old_z: the old bins array.
    :return: The new bins array (int8)
    """

    for i in range(1, len(z) - 1):

        z[i] = (quantums[i-1] + quantums[i]) / 2

    return np.round(z)
def calculate_quantum_values(hist_orig, z, quantums):
    """
    This function calculates the new values of the quantums
     array according to the current z bins.
    :param hist_orig:
    :param z: The bins array
    :return: The new quantums array (float64)
    """

    for i in range(len(quantums)):

        z_range = np.arange(z[i] + 1, z[i + 1] + 1)
        hist_z_range = hist_orig[z_range]
        hist_z_range_sum = hist_z_range.sum()

        quantums[i] = (z_range.dot(hist_z_range)) / hist_z_range_sum

    return  np.round(quantums)

def calculate_error(z, hist_orig, quantums, n_quants):
    """
    This function calculates the error for a given z and q arrays.
    :param z: the segments array
    :param hist_orig: the image histogram
    :param quantums: The current q values array
    :return: the current error calculated according to the given z and q
    values.
    """
    err = 0

    for i in range(n_quants):

        for j in range(z[i] + 1, z[i + 1] + 1):

            err += ((quantums[i] - j) ** 2) * hist_orig[j]

    err += (quantums[-1] - 255) ** 2 * hist_orig[255]

    return err
