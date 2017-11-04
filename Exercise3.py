# Philip Yuan
# EC601

import cv2
import numpy as np


# Gaussian noise
def noise_gaussian(image,mean,sigma):
    row, column = image.shape
    ga = np.random.normal(mean,sigma,(row,column))
    ga = ga.reshape(row,column)
    result = image + ga
    return result


# Salt and pepper noise
def noise_snp(image,prob_s,prob_p):
    result = np.copy(image)

    n_s = np.ceil(image.size * prob_s)
    c_s = [np.random.randint(0, i-1, int(n_s))
           for i in image.shape]
    result[c_s] = 255

    n_p = np.ceil(image.size * prob_p)
    c_p = [np.random.randint(0, j-1, int(n_p))
           for j in image.shape]
    result[c_p] = 0

    return result


# Smoothing filters
def smoothing(g_image, snp_image, kernel):
    # Original
    cv2.imwrite('./filter/g_original.png',g_image)
    cv2.imwrite('./filter/snp_original.png',snp_image)

    # Box filter
    g_box = cv2.blur(g_image,kernel)
    cv2.imwrite('./filter/g_box.png',g_box)
    snp_box = cv2.blur(snp_image,kernel)
    cv2.imwrite('./filter/snp_box.png',snp_box)

    # Gaussian filter
    g_gauss = cv2.GaussianBlur(g_image,kernel,0)
    cv2.imwrite('./filter/g_gauss.png',g_gauss)
    snp_gauss = cv2.GaussianBlur(snp_image,kernel,0)
    cv2.imwrite('./filter/snp_gauss.png',snp_gauss)

    # Median filter
    g_median = cv2.medianBlur(g_image.astype(np.uint8),kernel[0])
    cv2.imwrite('./filter/g_median.png',g_median)
    snp_median = cv2.medianBlur(snp_image.astype(np.uint8),kernel[0])
    cv2.imwrite('./filter/snp_median.png',snp_median)
    return


if __name__ == '__main__':
    lenna = cv2.imread('Lenna.png',cv2.IMREAD_GRAYSCALE)

    # Gaussian noise
    gaussian_0_0 = noise_gaussian(lenna,0,0)
    gaussian_5_20 = noise_gaussian(lenna,5,20)
    gaussian_10_50 = noise_gaussian(lenna,10,50)
    gaussian_20_0 = noise_gaussian(lenna,20,0)
    gaussian_20_20 = noise_gaussian(lenna,20,20)
    gaussian_20_50 = noise_gaussian(lenna,20,50)
    gaussian_20_100 = noise_gaussian(lenna,20,100)
    cv2.imwrite('./noise/gaussian_0_0.png',gaussian_0_0)
    cv2.imwrite('./noise/gaussian_5_20.png',gaussian_5_20)
    cv2.imwrite('./noise/gaussian_10_50.png',gaussian_10_50)
    cv2.imwrite('./noise/gaussian_20_0.png',gaussian_20_0)
    cv2.imwrite('./noise/gaussian_20_20.png',gaussian_20_20)
    cv2.imwrite('./noise/gaussian_20_50.png',gaussian_20_50)
    cv2.imwrite('./noise/gaussian_20_100.png',gaussian_20_100)

    # Salt and pepper noise
    snp_1_1 = noise_snp(lenna,0.01,0.01)
    snp_3_3 = noise_snp(lenna,0.03,0.03)
    snp_5_5 = noise_snp(lenna,0.05,0.05)
    snp_40_40 = noise_snp(lenna,0.40,0.40)
    cv2.imwrite('./noise/snp_1_1.png',snp_1_1)
    cv2.imwrite('./noise/snp_3_5.png', snp_3_3)
    cv2.imwrite('./noise/snp_5_5.png', snp_5_5)
    cv2.imwrite('./noise/snp_40_40.png',snp_40_40)

    # Do smoothing filtering for a set of gaussian/salt+pepper noise image
    smoothing(gaussian_20_20,snp_5_5,(7,7))
