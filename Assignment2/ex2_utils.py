import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
Author : Aiman Younis
ID : 207054354
"""
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
IMG_INT_MAX_VAL = 255


def myID() -> np.int:
    """
    Return my ID
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
    img = cv2.imread(filename, -1)
    if representation is LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to range [0,1]
    img = img.astype(np.float) / IMG_INT_MAX_VAL
    if representation is LOAD_GRAY_SCALE and len(img.shape) > 2:
        b, g, r = np.split(img, 3, axis=2)
        img = 0.3 * r + 0.59 * g + 0.11 * b
        img = img.squeeze()

    elif representation is LOAD_RGB and len(img.shape) < 3:
        img = np.stack((img, img, img), axis=2)

    return img


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    signal_len, kernel_len = inSignal.size, kernel1.size
    pad_signal = np.pad(inSignal, (kernel_len - 1,))  # padding with zeroes
    flip_kernel = np.flip(kernel1)  # flip the kernel
    # mode="full"-size of output vector is: signal_len + kernel_len - 1
    # multiply the kernel from i to i + kernel_size
    return np.array([np.dot(pad_signal[i:i + kernel_len], flip_kernel) for i in range(signal_len + kernel_len - 1)])


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    # get image and kernel shape
    image_height, image_width = inImage.shape[:2]
    kernel_height, kernel_width = kernel2.shape[:2]
    pad_image = image_padding(inImage, kernel_height, kernel_width)  # pad the image

    return np.array([np.sum(pad_image[i:i + kernel_height, j:j + kernel_width] * kernel2)
                     for i in range(image_height) for j in range(image_width)]).reshape((image_height,image_width))


def image_padding(image: np.ndarray, kernel_height: int, kernel_width: int) -> np.ndarray:
    """
    :param image: input image
    :param kernel_height: kernel height
    :param kernel_width: kernel width
    :return: padding image ’full’, ’borderType’=cv2.BORDER REPLICATE
    """
    pad_len = kernel_height // 2

    if kernel_width == kernel_height:
        pad_image = cv2.copyMakeBorder(image, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_REPLICATE)

    elif kernel_width > kernel_height:
        pad_len = kernel_width // 2
        pad_image = cv2.copyMakeBorder(image, 0, 0, pad_len, pad_len, cv2.BORDER_REPLICATE)
    else:
        pad_image = cv2.copyMakeBorder(image, pad_len, pad_len, 0, 0, cv2.BORDER_REPLICATE)

    return pad_image


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    # ker_x-kernel for derive according x, ker_y-kernel for derive according y
    ker_x = np.array([[-1, 0, 1]])
    # the y kernel is x kernel transposed
    ker_y = np.transpose(ker_x)
    # ker_x-derive according x, ker_y-derive according y
    # calculating the x and y derivatives
    x_der, y_der = conv2D(inImage, ker_x), conv2D(inImage, ker_y)
    # calu
    return np.arctan2(y_der, x_der), calc_magnitude(x_der, y_der), x_der, y_der


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation to get the difference
    """
    # cv implementation
    cv_sobel_x, cv_sobel_y = cv2.Sobel(img, -1, 1, 0, ksize = 3), cv2.Sobel(img, -1, 0, 1, ksize=3)
    # my implementation
    sobel_x, sobel_y = get_sobol_Ix_Iy(img)
    return (apply_threshold(calc_magnitude(cv_sobel_x, cv_sobel_y), thresh),
            apply_threshold(calc_magnitude(sobel_x, sobel_y), thresh))


def calc_magnitude(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    :param image1: first derive image
    :param image2: second derive image
    :return: magnitude image between image1 and image2
    """
    return np.sqrt(np.square(image1) + np.square(image2))


def apply_threshold(image: np.ndarray, threshold) -> np.ndarray:
    """
    image[i,j] = 1 if image[i,j] > threshold else zero
    :param image: input image
    :param threshold: threshold value to determine edge
    :return: binary image
    """
    threshold_image = np.zeros(image.shape)
    threshold_image[image > threshold] = 1
    return threshold_image


def get_sobol_Ix_Iy(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param image: blurred image
    :return: derivatives rows, cols
    """
    ker_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ker_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return conv2D(image, np.flip(ker_sobel_x)), conv2D(image, np.flip(ker_sobel_y))


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # blur the image with 2D gaussian kernel, apply Laplacian filter, and search zero crossing
    return zero_crossing(laplacian_conv(blurImage1(img, 7)))


def laplacian_conv(image: np.ndarray) -> np.ndarray:
    """
    :param image: input image
    :return: image after applying convolution with laplacian_filter
    """
    laplac_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # [0, 1, 0]
                                                                 # [1, -4, 1]
                                                                 # [0, 1, 0]
    return conv2D(image, np.flip(laplac_filter)) # return convolution with this


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    gaus_1d = get_gaus_1d(kernel_size)
    return conv2D(in_image, np.dot(gaus_1d, np.transpose(gaus_1d)))


def get_gaus_1d(kernel_size: int) -> np.ndarray:
    """
    my implementation to gaussian 1D kernel according opencv doc
    :param kernel_size: size of the kernel
    :return: gaussian 1D kernel
    The function of gaussian kernel is 1/2pi*o * e^-x^2/2o^2
    """
    gaus_1d = np.array([[np.exp(-(np.square(x - (kernel_size - 1) / 2)) / (2 * np.square(get_gaus_sigma(kernel_size))))
                         for x in range(kernel_size)]])
    gaus_1d /= gaus_1d.sum()
    return np.transpose(gaus_1d)


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    return cv2.filter2D(in_image, -1, get_gaussian2D(kernel_size, get_gaus_sigma(kernel_size)))


def get_gaus_sigma(kernel_size: int):
    """
    :param kernel_size: size of kernel
    :return: value of sigma factor in gaussian filter
    """
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8


def get_gaussian2D(size: int, sigma: float) -> np.ndarray:
    """
    :param size: kernel size, It should be odd.
    :param sigma: gaussian standard deviation.
    :return: gaussian 2D filter in shape ksize x ksize
    """
    gaus_kernel = cv2.getGaussianKernel(size, sigma)
    return np.dot(gaus_kernel, np.transpose(gaus_kernel))


def zero_neighbor_exist(image: np.ndarray, i: int, j: int) -> bool:
    """
    :param image: input image
    :param i: row to neighbor check
    :param j: col to neighbor check
    :return: true if exist neighbor with 0 value
    """
    neighbour = [image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1], image[i, j + 1], image[i + 1, j + 1],
                 image[i + 1, j], image[i + 1, j - 1], image[i, j - 1]]
    return min(neighbour) == 0


def zero_crossing(image: np.ndarray) -> np.ndarray:
    """
    search of templates like : {-,+} or {+,-}
    :param image: input image
    :return: binary zero-crossing image(=if zero-crossing value=1 else value=0)
    """
    img_height, img_width = image.shape  # get image shape
    bin_image = apply_threshold(image, 0)  # create a binary image

    zero_cros_bin = np.zeros((img_height, img_width))
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if bin_image[i, j] > 0 and zero_neighbor_exist(bin_image, i, j):
                zero_cros_bin[i, j] = 1

    return zero_cros_bin


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, and Canny from scratch implementation
    """
    i_x, i_y = get_sobol_Ix_Iy(img)
    mag, deg = calc_magnitude(i_x, i_y), angle_in_degrees_0_180(i_y, i_x)
    return (cv2.Canny((img * 255).astype("uint8"), thrs_1, thrs_2),
            hysteresis(non_maximum_suppression(mag, direction_quantization(deg)), thrs_1 / 255, thrs_2 / 255))


def angle_in_degrees_0_180(i_x: np.ndarray, i_y: np.ndarray) -> np.ndarray:
    """
    calculate the angle in degrees [0,180] between two arrays
    :param i_x: first array
    :param i_y: second array
    :return: array of degrees between [0,180]
    """
    # computing the direction of the gradient
    return np.mod(np.rad2deg(np.arctan2(i_y, i_x)), 180)


def direction_quantization(directions: np.ndarray) -> np.ndarray:
    """
    explain : this function is to quantize the gradient direction
    """
    quant_deg = np.zeros(directions.shape)
    quant_deg[np.logical_or(np.logical_and(0 <= directions, directions > 22.5),
                            np.logical_and(157.5 <= directions, directions >= 180))] = 0
    quant_deg[np.logical_and(22.5 <= directions, directions > 67.5)] = 45
    quant_deg[np.logical_and(67.5 <= directions, directions > 112.5)] = 90
    quant_deg[np.logical_and(112.5 <= directions, directions > 157.5)] = 135
    return quant_deg


def non_maximum_suppression(magnitude: np.ndarray, quant_deg: np.ndarray) -> np.ndarray:
    """
    explain : we convert the blured edges to sharp edges , by preserving the local maxima in gradient image
    the indexes of [i - 1 , j] [i + 1 ,j ] to compare the edge strength
    :param magnitude: magnitude if an image (gradient)
    :param quant_deg: direction of the gradient
    :return: thin_edge
    """
    height, width = magnitude.shape[:2]  # gets magnitude shape
    thin_edge = np.zeros((height, width))  # output

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            curr_deg = quant_deg[i, j]
            if curr_deg == 0:
                thin_edge[i, j] = calc_thin_mag(magnitude[i, j], magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif curr_deg == 45:
                thin_edge[i, j] = calc_thin_mag(magnitude[i, j], magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif curr_deg == 90:
                thin_edge[i, j] = calc_thin_mag(magnitude[i, j], magnitude[i, j - 1], magnitude[i, j + 1])
            else:  # curr_deg == 135
                thin_edge[i, j] = calc_thin_mag(magnitude[i, j], magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])

    return thin_edge


def calc_thin_mag(curr_pixel: float, first_pixel: float, second_pixel: float) -> float:
    """
    :param curr_pixel: current pixel intensity
    :param first_pixel: first pixel intensity
    :param second_pixel: second pixel intensity
    :return: if the max value is the curr_pixel return  curr_pixel's intensity otherwise 0 means we suppress the value
    or we delete it.
    """
    if max(curr_pixel, first_pixel, second_pixel) == curr_pixel:
        return curr_pixel

    return 0


def hysteresis(thin_edges: np.ndarray, thrs_1: float, thrs_2: float) -> np.ndarray:
    """
    :explain : every pixel is connected to edge pixel and bigger than threshold 2 it's also a edge pixel.
    :param thin_edges:
    :param thrs_1: this is the first threshold
    :param thrs_2: the second threshold
    :return: image.
    """
    img_thrs_1 = apply_threshold(thin_edges, thrs_1)
    height, width = img_thrs_1.shape[:2]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if img_thrs_1[i, j] == 1:  # if the current value bigger then thrs_1
                # Take the neighbors and check if the value of them is greater than greater than threshold_2
                img_thrs_1[i - 1:i + 2, j - 1: j + 2] = neighbors_above_thrsh2(thin_edges[i - 1:i + 2, j - 1: j + 2],
                                                                               thrs_2)

    return img_thrs_1


def neighbors_above_thrsh2(neighbours: np.ndarray, thrs_2: float) -> np.ndarray:
    """
    :param neighbours: neighbours of pixel i,j
    :param thrs_2: threshold value
    :return: position (i,j) = 1 iff value(i,j) > thrs_2
    """
    neighbours[neighbours > thrs_2] = 1
    return neighbours


def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    circles_list = []
    _,direct = get_sobol_Ix_Iy(img)
    direct = np.radians(direct)
    any_circle = np.zeros((len(img),len(img[0]),max_radius+1))
    canny_img = cv2.Canny(img,50,100)
    for x in range(0,len(canny_img)):
        for y in range(0,len(canny_img[0])):
            if canny_img[x][y] > 0:
                for r in range(min_radius,max_radius+1):
                    cy1 = int(y + r * np.sin(direct[x, y]-np.pi/2))
                    cx1 = int(x - r * np.cos(direct[x, y]-np.pi/2))
                    cy2 = int(y - r * np.sin(direct[x, y]-np.pi/2))
                    cx2 = int(x + r * np.cos(direct[x, y]-np.pi/2))
                    if 0 < cx1 < len(any_circle) and 0 < cy1 < len(any_circle[0]):
                        any_circle[cx1,cy1,r] += 1
                    if 0 < cx2 < len(any_circle) and 0 < cy2 < len(any_circle[0]):
                        any_circle[cx2, cy2, r] += 1

    # filtering with thresh
    thresh = 0.50*any_circle.max()
    b_center,a_center,radius=np.where(any_circle >= thresh)

    # deleting a similar circles
    eps = 14
    for j in range(0, len(a_center)):
        if a_center[j] == 0 and b_center[j] == 0 and radius[j] == 0:
            continue
        temp = (b_center[j], a_center[j], radius[j])
        index_a = np.where((temp[0]-eps <= b_center) & (b_center <= temp[0]+eps)
                           & (temp[1]-eps <= a_center) & (a_center <= temp[1]+eps)
                           & (temp[2]-eps <= radius) & (radius <= temp[2]+eps))[0]
        for i in range(1, len(index_a)):
            b_center[index_a[i]] = 0
            a_center[index_a[i]] = 0

            radius[index_a[i]] = 0

    # building the last list
    for i in range(0, len(a_center)):
        if a_center[i] == 0 and b_center[i] == 0 and radius[i] == 0:
            continue
        circles_list.append((a_center[i], b_center[i], radius[i]))

    return circles_list