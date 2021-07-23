import numpy as np
import matplotlib.pyplot as plt
import cv2
"""
@Author : Aiman Younis
Comments : I will submit my main , please run my main in order to see the answers.
Thank You shai :))
"""
def ID() -> int:
    return 207054354
pass

def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    Find the disparity matrix that represent the differences of positions between left image and right image.
    disp_map[i][j] = J_R - J_L
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    we want to compute the ssd so we get the kernel size then we will pass the kernel and find the coordinates that give
    us the best ssd value
    """
    kernel_half = int ((k_size*2 + 1) //2)
    w , h = img_r.shape
    # the depth of the image
    depth = np.zeros((w , h))
    for y in range (kernel_half, (w - kernel_half)): # iterate through the rows
        for x in range(kernel_half, (h - kernel_half)): # iterate through the columns
            best_offset = 0
            pixel = 0
            prev_ssd = 654354
            for offset in range(disp_range[0], disp_range[1]): # check the kernel which is exit in this range
                ssd = 0
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half , kernel_half):
                        # calculate the difference between the left and right kernel and then make the disp point to be
                        # the the offset with the minimum SSD (Sum of square difference)
                        # arg_min =>(I_left(x , y) - I_right (x + v, y +u))^2
                        ssd += (img_r [y+v, x+u] - img_l[(y + v), (x + u) - offset])**2
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            depth[y, x] = best_offset

    print(depth)

    return depth
    pass

def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    h, w = img_l.shape
    depth = np.zeros((h ,w))
    kernel_half = int ((k_size*2 +1) // 2)
    w, h = img_r.shape
    for y in range(kernel_half , w - kernel_half):
        for x in range(kernel_half , h - kernel_half):
            pred_ncc = 0
            best_offset = 0
            sum1 = 0 # I1 *I2
            sum2 = 0.0001 # (I1)^2 the right image multiplied
            sum3 = 0.0001  # (I2)^2 the left image multiplied
            mean1 = 0
            mean2 = 0
            n = 0
            for offset in range(disp_range[0], disp_range[1]):
                ncc = 0
                for u in range(-kernel_half , kernel_half):
                    for v in range(-kernel_half , kernel_half):
                        mean1 += img_r [y + v, x + u]
                        mean2 += img_l[y + v, (x + u) - offset]
                        n += 1
                mean1 = mean1  /n
                mean2 = mean2 / n
                for u in range(-kernel_half, kernel_half):
                    for v in range(-kernel_half, kernel_half):
                        sum1 += np.dot(((img_r[y + v, x + u]) - mean1) , ((img_l[y + v, (x + u) - offset]) - mean2))
                        sum2 += ((img_r[y + v, x + u]) - mean1)**2
                        sum3 += ((img_l[y + v, x + u]) - mean2)**2

                ncc = sum1 / np.sqrt(np.dot(sum2 , sum3))
                if ncc > pred_ncc:
                    pred_ncc = ncc
                    best_offset = offset
            depth[y, x] =  best_offset

    return depth
    pass

def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """

    A = []
    for i in range(0, len(src_pnt)):
        x, y = src_pnt[i][0], src_pnt[i][1]
        u, v = dst_pnt[i][0], dst_pnt[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])# like we saw in class append for evey point two rows
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A) # use SVD to find the values of the variables in the matrix
    L = Vh[-1, :] / Vh[-1, -1]  # divided by the last row like we see in the exercise
    H = L.reshape(3, 3) # reshaping to 3 by 3
    print(H) # print our Homography
    #print openCv homography
    M, mask = cv2.findHomography(src_pnt, dst_pnt)
    print("=======================")
    print(M)
    return H

pass


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()
    size = src_img.shape
    # no need to take the coordinates of the second image in order to do the homography just pick the corners
    # coordinates
    pts_src = np.array(
        [
            [0, 0],
            [size[1] - 1, 0],
            [size[1] - 1, size[0] - 1],
            [0, size[0] - 1]
        ], dtype=float
    )
    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    h = computeHomography(pts_src, dst_p) # my function to find the homography matrix in order to do projection
    # to the coordinates by this equations from opencv dst(x,y) = src(m11x + m12y +m13/ m31x +m32y +m33
    # , m21x + m22y +m23/ m31x +m32y +m33)
    im_temp = warpPerspective(src_img , h, (dst_img.shape[1],dst_img.shape[0]))
    plt.imshow(im_temp)
    plt.show()
    im_dst2 =  im_temp + dst_img
    plt.imshow(im_dst2.astype('uint8'))
    plt.show()

    pass


def warpPerspective(img, M, dsize):
    """

    :param img:
    :param M:
    :param dsize:
    :return: return the projection of the image using the homography matrix by this equation:
    dst(x,y) = src(m11x + m12y +m13/ m31x +m32y +m33  , m21x + m22y +m23/ m31x +m32y +m33)
    """
    mtr = to_mtx(img)
    R, C = dsize
    dst = np.zeros((R, C, mtr.shape[2]))
    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.dot(M, [i, j, 1])
            i2, j2, _ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[i2, j2] = mtr[i, j]

    return to_img(dst)


def to_mtx(img):
    """
    :param img:  converts the image to matrix
    :return:
    """
    H, V, C = img.shape
    mtr = np.zeros((V, H,C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]

    return mtr


def to_img(mtr):
    """
    :param mtr:
    :return: converts the image to matrix
    """
    V, H, C = mtr.shape
    img = np.zeros((H, V, C), dtype='int')
    for i in range(mtr.shape[0]):
        img[:, i] = mtr[i]

    return img