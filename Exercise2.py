import cv2
import numpy as np

lenna_bgr = cv2.imread('Lenna.png',cv2.IMREAD_COLOR)
bgr_b, bgr_g, bgr_r = cv2.split(lenna_bgr)
cv2.imwrite('./colorspaces/bgr_b.png',bgr_b)
cv2.imwrite('./colorspaces/bgr_g.png',bgr_g)
cv2.imwrite('./colorspaces/bgr_r.png',bgr_r)
pixel_bgr = lenna_bgr[20,25]

lenna_ycrcb = cv2.cvtColor(lenna_bgr,cv2.COLOR_BGR2YCR_CB)
ycrcb_y, ycrcb_cr, ycrcb_cb = cv2.split(lenna_ycrcb)
cv2.imwrite('./colorspaces/ycrcb_y.png', ycrcb_y)
cv2.imwrite('./colorspaces/ycrcb_cr.png', ycrcb_cr)
cv2.imwrite('./colorspaces/ycrcb_cb.png', ycrcb_cb)
pixel_ycrcb = lenna_ycrcb[20,25]

lenna_hsv = cv2.cvtColor(lenna_bgr, cv2.COLOR_BGR2HSV)
hsv_h, hsv_s, hsv_v = cv2.split(lenna_hsv)
cv2.imwrite('./colorspaces/hsv_h.png', hsv_h)
cv2.imwrite('./colorspaces/hsv_s.png', hsv_s)
cv2.imwrite('./colorspaces/hsv_v.png', hsv_v)
pixel_hsv = lenna_hsv[20,25]

print('Pixel values at [20,25]')
print('BGR: ', pixel_bgr)
print('YCrCb: ', pixel_ycrcb)
print('HSV: ', pixel_hsv)