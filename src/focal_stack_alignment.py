import os
import cv2
import scipy.ndimage
import numpy as np

from scipy.interpolate import RectBivariateSpline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_dir = '../data'
# item_name = '07_small'
# item_name = '07_tiny'
# item_name = '05_small'
item_name = 'mobo2_small2'

result_dir = '../result'


def inverse_compositional_image_alignment(It, It1, rect, thresh=.025, maxIt=100):
    '''
   Matthew-Bakers Inverse Compositional Alignment with Affine Matrix

      Inputs: 
        It: template image
        It1: Current image
        rect: Current position of the object
        (top left, bottom right coordinates, x1, y1, x2, y2)
        thresh: Stop condition when dp is too small
        maxIt: Maximum number of iterations to run

      Outputs:
        M: Affine mtarix (2x3)
    '''

    # Set thresholds (you probably want to play around with the values)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    x1, y1, x2, y2 = rect
    maxIter = maxIt
    dp = np.array([float("inf")] * 6) 

    interpolated_It = RectBivariateSpline(
        x=np.array([i for i in range(int(It.shape[0]))]),
        y=np.array([i for i in range(int(It.shape[1]))]),
        z=It
    )
    interpolated_It1 = RectBivariateSpline(
        x=np.array([i for i in range(int(It1.shape[0]))]),
        y=np.array([i for i in range(int(It1.shape[1]))]),
        z=It1
    )
    x = np.arange(x1, x2 + .5)
    y = np.arange(y1, y2 + .5)
    X = np.array([x for i in range(len(y))])
    Y = np.array([y for i in range(len(x))]).T
    Itx = interpolated_It.ev(Y, X)

    dx = interpolated_It.ev(Y, X, dx=0, dy=1).flatten()
    dy = interpolated_It.ev(Y, X, dx=1, dy=0).flatten()

    A = np.array([
        dx * X.flatten(),
        dx * Y.flatten(),
        dx,
        dy * X.flatten(),
        dy * Y.flatten(),
        dy,
    ]).T

    for i in range(maxIter):
        if np.sum(np.linalg.norm(dp)) < thresh:
            break
        warp_X = p[0] * X + p[1] * Y + p[2]
        warp_Y = p[3] * X + p[4] * Y + p[5]
        valid_points = (warp_X >= x1) & (warp_X <= x2) & (
            warp_Y >= y1) & (warp_Y <= y2)
        warp_X, warp_Y = warp_X[valid_points], warp_Y[valid_points]
        warp_It1x = interpolated_It1.ev(warp_Y, warp_X)

        A_valid = A[valid_points.flatten()]
        b = (warp_It1x - Itx[valid_points]).flatten()

        dp = np.dot(np.linalg.inv(np.dot(A_valid.T, A_valid)),
                    np.dot(A_valid.T, b))

        M = np.copy(p).reshape(2, 3)
        M = np.vstack((M, np.array([[0, 0, 1]])))
        dM = np.vstack((np.copy(dp).reshape(2, 3), np.array([[0, 0, 1]])))
        dM[0, 0] += 1
        dM[1, 1] += 1
        M = np.dot(M, np.linalg.inv(dM))
        p = M[:2, :].flatten()

    M = M[:2]
    return M, i, np.sum(np.linalg.norm(dp))


def lukas_kanade(It, It1, rect, thresh=.001, maxIt=100):
    '''
    Lucas-Kanade Forward Additive Alignment with Translation Only

      Inputs: 
        It: template image
        It1: Current image
        rect: Current position of the object
        (top left, bottom right coordinates, x1, y1, x2, y2)
        thresh: Stop condition when dp is too small
        maxIt: Maximum number of iterations to run

      Outputs:
        p: movement vector dx, dy
    '''

    p = np.zeros(2)  # dx, dy
    threshold = thresh
    maxIters = maxIt
    x1, y1, x2, y2 = rect

    dp = np.array([float("inf"), float("inf")])
    interpolated_It = RectBivariateSpline(
        x=np.array([i for i in range(int(It.shape[0]))]),
        y=np.array([i for i in range(int(It.shape[1]))]),
        z=It
    )
    interpolated_It1 = RectBivariateSpline(
        x=np.array([i for i in range(int(It1.shape[0]))]),
        y=np.array([i for i in range(int(It1.shape[1]))]),
        z=It1
    )

    x = np.arange(x1, x2 + .5)
    y = np.arange(y1, y2 + .5)
    X = np.array([x for i in range(len(y))])
    Y = np.array([y for i in range(len(x))]).T
    Itx = interpolated_It.ev(Y, X)

    for i in range(maxIters):
        if np.sum(np.linalg.norm(dp)) < threshold:
            break
        warp_x = np.arange(x1 + p[0], x2 + p[0] + .5)
        warp_y = np.arange(y1 + p[1], y2 + p[1] + .5)
        warp_X = np.array([warp_x for i in range(len(warp_y))])
        warp_Y = np.array([warp_y for i in range(len(warp_x))]).T
        warp_It1x = interpolated_It1.ev(warp_Y, warp_X)

        A = np.array([
            interpolated_It1.ev(warp_Y, warp_X, dx=0, dy=1).flatten(),
            interpolated_It1.ev(warp_Y, warp_X, dx=1, dy=0).flatten()
        ]).T
        b = (Itx - warp_It1x).flatten()

        dp = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        p += dp

    return p, i, np.sum(np.linalg.norm(dp))


def all_in_focus_image_stitching():
    
    pass


def main():

    img_dir = os.path.join(data_dir, item_name, 'images')
    img_paths = []
    for img_name in os.listdir(img_dir):
        img_paths.append(os.path.join(img_dir, img_name))

    last_affine_M = np.identity(3).astype(np.float32)
    last_trans_p = np.zeros(2)

    result_img_dir = os.path.join(result_dir, item_name, 'images')
    if not os.path.exists(result_img_dir):
        os.makedirs(result_img_dir)

    for i in range(len(img_paths) - 1):
        template_img = cv2.imread(img_paths[i])
        input_img = cv2.imread(img_paths[i + 1])

        template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        rect = np.array([0, 0, 960, 640])

        affine_M, iter_num, err = inverse_compositional_image_alignment(
            input_img_gray, template_img_gray, rect, thresh=1e-5)
        print("idx = {}, iter = {}, error = {}".format(i, iter_num, err))

        aligned_img_gray = cv2.warpAffine(
            input_img_gray, affine_M, dsize=(input_img.shape[1], input_img.shape[0]))

        affine_M = np.concatenate(
            (affine_M, np.array([[0, 0, 1]], dtype=np.float32)), axis=0).dot(last_affine_M)
        last_affine_M = affine_M

        # trans_p = lukas_kanade(
        #     template_img_gray, aligned_img_gray, rect) + last_trans_p
        # last_trans_p = trans_p

        # trans_M = np.array(
        #     [[1.0, 0.0, trans_p[0]], [0.0, 1.0, trans_p[1]], [0.0, 0.0, 1.0]])
        
        # M = (trans_M @ affine_M)[:2]
        M = affine_M[:2]

        aligned_img = cv2.warpAffine(input_img, M, dsize=(
            input_img.shape[1], input_img.shape[0]))
        cv2.imwrite(os.path.join(result_img_dir,
                    '{:02d}.png'.format(i + 1)), aligned_img)


if __name__ == '__main__':
    main()
