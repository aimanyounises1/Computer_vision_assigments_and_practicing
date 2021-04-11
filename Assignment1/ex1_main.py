from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time

from Assigment1.ex1_utils import imDisplay, imReadAndConvert, transformRGB2YIQ, LOAD_RGB, LOAD_GRAY_SCALE, myID, \
    quantizeImage, hsitogramEqualize


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 3, 10)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    plt.gray()
    plt.imshow(img_lst)
    plt.figure()
    plt.imshow(img_lst)

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'ameen.JPGq'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    imDisplay(img_path, LOAD_RGB)
    # Convert Color spaces to RGB
    img = imReadAndConvert(img_path, LOAD_RGB)
    ###
    f,ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    yiq_img = transformRGB2YIQ(img)
    ax[1].imshow(yiq_img)
    plt.show()

    # Image histEq
    print("The hist Demo 1")
    histEqDemo(img_path, LOAD_GRAY_SCALE)
    print("The hist Demo 2")
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    print("The quantzation")
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
