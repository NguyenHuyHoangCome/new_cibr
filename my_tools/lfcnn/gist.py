import numpy as np
import numpy.matlib as nm
import numpy.fft as f
from PIL import Image
import cv2
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi


def compute_gist_descriptor(imgGray):
    # build average feature map:
    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        data = np.real(kernel).reshape((np.real(kernel).shape[0], np.real(kernel).shape[1], 1))
        return np.sqrt(ndi.convolve(image, data, mode='wrap') ** 2 + ndi.convolve(image, data, mode='wrap') ** 2)

    def make_square(img):

        r, c, channels = img.shape

        side4 = (int(min([r, c]) / 4)) * 4

        one_edge = int(side4 / 2)
        r2 = int(r / 2)
        c2 = int(c / 2)

        img1 = img[((r2) - one_edge):((r2) + one_edge), ((c2) - one_edge):((c2) + one_edge)]

        r, c, channels = img1.shape
        return img1[:min([r, c]), :min([r, c])]

    def compute_avg(img):
        img = make_square(img)

        r, c, channels = img.shape

        chunks_row = np.split(np.array(range(r)), 4)
        chunks_col = np.split(np.array(range(c)), 4)

        grid_images = []

        for row in chunks_row:
            for col in chunks_col:
                grid_images.append(np.mean(img[np.min(row):np.max(row), np.min(col):np.max(col)]))
        return np.array(grid_images).reshape((4, 4))

    def power_single(gs):
        (kernel, powers) = gs
        return powers * 255
    
    
    images = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)

    images = make_square(images)
    images = images / 255.0

    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []

    for theta in (0, 1, 2, 3, 4, 5, 6, 7):
        theta = theta / 8. * np.pi
        for frequency in (0.1, 0.2, 0.3, 0.4):
            # for frequency in (0.1, 0.2, 0.4, 0.6, 0.8):
            kernel = gabor_kernel(frequency, theta=theta)

            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, power(images, kernel)))
    return np.array([compute_avg(power_single(img)) for img in results]).reshape(512, )
