import cv2
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.matlab_functions import imresize

gt = cv2.imread('test_images/7.png')
lam = cv2.imread('Results/ECCV2_Ours_7.png')

img = lam[:, -256:, :]
# print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey(0)

psnr = calculate_psnr(gt, img, crop_border=0, input_order='HWC')
ssim = calculate_ssim(gt, img, crop_border=0, input_order='HWC')
print(psnr, ssim)