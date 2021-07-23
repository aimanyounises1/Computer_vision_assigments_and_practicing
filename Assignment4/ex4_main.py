# ps2
import os
import numpy as np
from ex4_utils import *
import cv2

from Assignment4.ex4_utils import disparitySSD, disparityNC, warpImag, ID


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd,0)
    plt.colorbar()
    plt.show()


def main():
    ## 1-a
    # Read images
    i = 1
    id  = ID()
    print(id)
    L = cv2.imread('pair0-R.png', 0) / 255
    R = cv2.imread('pair0-L.png', 0) / 255
    # cv2.imshow("L",L)
    # cv2.waitKey(0)
    # cv2.imshow("R",R)
    # cv2.waitKey(0)

    # Display depth SSD
    displayDepthImage(L, R, (0, 5), method = disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (0, 5), method = disparityNC)

    src = np.array([[279, 552],
                     [372, 559],
                     [362, 472],
                     [277, 469]])
    dst = np.array([[24, 566],
                     [114, 552],
                  [106, 474],
                     [19, 481]])
    h = computeHomography(src, dst)
    print(h)
   # h_src = Homogeneous(src)
    #pred = h.dot(h_src.T).T

    #pred = unHomogeneous(pred)
    #print(np.sqrt(np.square(pred-dst).mean()))
    # run this test , please don't normalize the second picture because the wrap perspective will not work
    dst = cv2.imread('billBoard.jpg')
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    src = cv2.imread('car.jpg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


    warpImag(src, dst)


if __name__=='__main__':
    main()