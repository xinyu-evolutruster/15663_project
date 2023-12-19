import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter as gaussian_filter
from scipy.optimize import least_squares as least_squares
from scipy.optimize import leastsq as leastsq

result_dir = '../result'
item_name = 'mobo2_tiny'

iter = 0

def create_disc_psf(radius):
    """
    Create a constant disc point spread function (PSF).
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disc = x**2 + y**2 <= radius**2
    disc = disc.astype(float)
    return disc / disc.sum()


def blur_image(image, psf):
    """
    Blur the image using the specified point spread function (PSF).
    """
    blurred_image = convolve(image, psf, mode='constant', cval=0.0)
    return blurred_image


def create_blur_stack(img, image_stack, rs, vis=False):
    height, width, num_frames = image_stack.shape
    blur_stack = np.zeros((height, width, len(rs)))

    i = 0
    for psf_radius in rs:
        psf = create_disc_psf(psf_radius)

        blur_stack[:, :, i] = blur_image(img, psf)
        i += 1

        # test
        if vis:
            fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
            axes[0].imshow(img)
            axes[1].imshow(blur_stack[:, :, i])
            plt.show()

        # test
        print(f"i = {i}, psf_radius={psf_radius}")
    
    return blur_stack


def compute_difference_map(image_stack, blur_stack, sigma=1):
    height, width, num_frames = image_stack.shape
    num_blur_level = blur_stack.shape[2]
    
    diff_maps = np.zeros((height, width, num_frames, num_blur_level), dtype=np.float32)

    for i in range(num_frames):
        frame = image_stack[:, :, i]
        for r in range(num_blur_level):
            diff_map = np.abs(frame - blur_stack[:, :, r])
            diff_map = gaussian_filter(diff_map, sigma=sigma, mode='constant', cval=0)
            diff_maps[:, :, i, r] = diff_map

    return diff_maps


def compute_blur_map(diff_maps, rs, sigma=1.0):
    height, width, num_frames, num_blur_level = diff_maps.shape

    blur_map = np.zeros((height, width, num_frames))

    for i in range(num_frames):
        diff_map = diff_maps[:, :, i, :]
        idx = np.argmin(diff_map, axis=2)
        blur_map[:, :, i] = rs[idx]

    return blur_map


def compute_confidence_map(diff_maps, rs, alpha=1.0):
    height, width, num_frames, num_blur_level = diff_maps.shape
    
    confidence_map = np.zeros((height, width, num_frames))

    for i in range(num_frames):
        diff_map = diff_maps[:, :, i, :]
        mean_map = np.mean(diff_map, axis=2)
        min_map = np.min(diff_map, axis=2)
        confidence_map[:, :, i] = (mean_map - min_map) ** alpha

    return confidence_map


def compute_bi(s, fi, F=2.0, A=3.0):
    # s is the depth map

    height, width = s.shape

    fi_map = np.ones_like(s).astype(np.float32) * fi
    F_map = np.ones_like(s).astype(np.float32) * F
    A_map = np.ones_like(s).astype(np.float32) * A

    return A_map * abs(fi_map - s) / s * F_map / (fi_map - F_map)


def levenberg_marquardt(x, blur_map, confidence_map):
    """
    For Levenberg-Marquardt, the nearest and
    farthest depths are set to 10 and 32. The focal length and
    aperture are set to 2 and 3.
    """
    num_frames = blur_map.shape[2]

    A = x[0]
    F = x[1]
    fis = x[2:2+num_frames]
    s = x[2+num_frames:]

    # print("A = {}, F = {}".format(A, F))

    num_frames = blur_map.shape[2]

    target = []
    s_reshaped = s.reshape(40, 60)

    for i in range(num_frames):
        fi = fis[i]
        bi = compute_bi(s_reshaped, fi, F, A)
        target.append((bi - blur_map[:, :, i]) * confidence_map[:, :, i])

    target = np.array(target).reshape(-1)

    # print("iter = {}".format(iter))
    # iter += 1

    return target


def main():
    # Load the all-in-focus image
    I_path = os.path.join(result_dir, item_name, 'lambda_0/stitched_images/stitched_img_0.png')
    img = cv2.imread(I_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the focus stack
    img_stack = []

    data_dir = os.path.join(result_dir, item_name, 'images')

    # i = 0
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)

        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[0:, 0:] / 255
        img = cv2.resize(img, (60, 40))
        # plt.imshow(img)
        # plt.show()
        img_stack.append(img)
        
        # i += 1
        # if i > 35:
        #     break

    img_stack = np.transpose(img_stack, (1, 2, 0))
    img_stack = np.array(img_stack)[:, :, 5:35]

    height, width, num_frames = img_stack.shape
    print("height = {}, width = {}, num_frames = {}".format(height, width, num_frames))

    # Create the blur stack
    num_blur_level = 30
    rs = np.array([1 + i * 0.05 for i in range(num_blur_level)])
    blur_stack = create_blur_stack(img, img_stack, rs)

    # Create the difference map
    diff_map = compute_difference_map(img_stack, blur_stack)

    # Create the blur map
    blur_map = compute_blur_map(diff_map, rs)

    # Create the confidence map
    confidence_map = compute_confidence_map(diff_map, rs)

    # Jointly optimize for aperture size, focal depths, focal length, and a depth map
    # For Levenberg-Marquardt, the nearest and farthest depths are set to 10 and 32. 
    fis = np.array([i for i in range(10, 10 + num_frames)])
    A = np.array([3.0])
    F = np.array([2.0])
    init_depth_map = np.ones((height, width), dtype=np.float32).reshape(-1)
    
    init_x = np.concatenate((A, F, fis, init_depth_map))

    print("init depth map shape: ", init_depth_map.shape)
    
    result = least_squares(
        levenberg_marquardt, 
        init_x, 
        # method='lm',
        # max_nfev=int(20),
        verbose=2,
        args=(blur_map.astype(np.float32), confidence_map.astype(np.float32))
    )

    # depth_map, cov_x, infodict, mesg, ier = leastsq(
    #     levenberg_marquardt, 
    #     init_depth_map, 
    #     # method='lm', 
    #     # max_nfev=int(1e9),
    #     args=(fis, blur_map.astype(np.float32), confidence_map.astype(np.float32))
    # )

    print("after least square")

    # depth_map = depth_map.reshape(height, width)
    depth_map = result.x[2+num_frames:]
    depth_map = depth_map.reshape(height, width)

    plt.imshow(depth_map, cmap='gray')
    plt.colorbar()
    plt.savefig('./depth_map.png')
    # plt.show()

    fig=plt.figure()
    ax = plt.axes()

    im = ax.imshow(depth_map, cmap='gray')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.savefig('./depth_map.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == '__main__':
    main()