import numpy as np
import cv2

def Angle_Image(gray):
    # Tính Angle Image
    def sHalf(T, sigma):
        temp = -np.log(T) * 2 * (sigma ** 2)
        return np.round(np.sqrt(temp))

    def calculate_filter_size(T, sigma):
        return 2 * sHalf(T, sigma) + 1

    def MaskGeneration(T, sigma):
        N = calculate_filter_size(T, sigma)
        shalf = sHalf(T, sigma)
        y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), range(-int(shalf), int(shalf) + 1))
        return x, y

    def calculate_gradient_X(x, y, sigma):
        temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
        return -((x * np.exp(-temp)) / sigma ** 2)

    def calculate_gradient_Y(x, y, sigma):
        temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
        return -((y * np.exp(-temp)) / sigma ** 2)

    def Create_Gx(fx, fy):
        gx = calculate_gradient_X(fx, fy, sigma)
        gx = (gx * 255)
        return np.around(gx)

    def Create_Gy(fx, fy):
        gy = calculate_gradient_Y(fx, fy, sigma)
        gy = (gy * 255)
        return np.around(gy)

    def smooth(img, kernel=None):
        if kernel is None:
            mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:
            mask = kernel
        i, j = mask.shape
        output = np.zeros((img.shape[0], img.shape[1]))
        image_padded = pad(img, mask)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                output[x, y] = (mask * image_padded[x:x + i, y:y + j]).sum() / mask.sum()
        return output

    def pad(img, kernel):
        #print(img.shape)
        r, c = img.shape
        kr, kc = kernel.shape
        padded = np.zeros((r + kr, c + kc), dtype=img.dtype)
        insert = np.uint((kr) / 2)
        padded[insert: insert + r, insert: insert + c] = img
        return padded

    def Gaussian(x, y, sigma):
        temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
        return (np.exp(-temp))

    def ApplyMask(image, kernel):
        i, j = kernel.shape
        kernel = np.flipud(np.fliplr(kernel))
        output = np.zeros_like(image)
        image_padded = pad(image, kernel)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                output[x, y] = (kernel * image_padded[x:x + i, y:y + j]).sum()
        return output

    def Gradient_Direction(fx, fy):
        g_dir = np.zeros((fx.shape[0], fx.shape[1]))
        g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
        return g_dir

    sigma = 0.5
    T = 0.3
    x, y = MaskGeneration(T, sigma)
    gauss = Gaussian(x, y, sigma)
    gx = -Create_Gx(x, y)
    gy = -Create_Gy(x, y)
    smooth_img = smooth(gray, gauss)
    fx = ApplyMask(smooth_img, gx)
    fy = ApplyMask(smooth_img, gy)
    Angle = Gradient_Direction(fx, fy)
    return Angle

def get_edh_37(img_input):
    if len(img_input)==3:
        Ig=cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    else:
        Ig=img_input.copy()
    Ie = cv2.Canny(Ig, 0, 255)
    w=Ie.shape[1]
    h=Ie.shape[0]
    
    Angle = Angle_Image(Ig)

    vEDH = np.zeros(37, dtype='float32')
    # tính tổng số điểm không phải biên
    sum = 0
    for i in range(h):
        for j in range(w):
            if Ie[i][j] == 0:
                sum = sum + 1
    vEDH[36] = sum / (w * h)

    # Tính số điểm biên của toàn ảnh
    total_edge_pixel = w * h - sum

    # tinh số lượng điểm biên của từng bin
    for i in range(h):
        for j in range(w):
            if Ie[i][j] == 0:
                continue
            # xác định số bin của điểm ảnh tọa độ (i,j)
            bin_index = int(Angle[i,j]/10)
            if bin_index==36:
                bin_index=0

            vEDH[bin_index] = vEDH[bin_index] + 1
            # print(vEDH[bin_index])
     # chuẩn hóa các his của từng bin
    for k in range(36):
        if total_edge_pixel > 0:
            vEDH[k] = vEDH[k] / total_edge_pixel
    return vEDH