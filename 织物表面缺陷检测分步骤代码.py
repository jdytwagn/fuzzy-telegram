import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
#读取图像
img = cv.imread('1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 显示原始图像
plt.subplot(331), plt.imshow(img,'gray'), plt.title('yst')
plt.axis('off')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(332), plt.imshow(img, 'gray'), plt.title('yshd')
plt.axis('off')
#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#设置高通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)
plt.subplot(333), plt.imshow(iimg, 'gray'), plt.title('flybh')
plt.axis('off')
new_img = np.zeros((512, 512), dtype = np.uint8)
for i in range(1, len(iimg)-1):
    for j in range(1, len(iimg[i])-1):
        sum = np.abs(iimg[i][j]-(iimg[i-1][j-1] + iimg[i-1][j] +
                    iimg[i-1][j+1] + iimg[i][j-1] +
                    iimg[i][j] + iimg[i][j+1] + iimg[i+1][j-1]
                    + iimg[i+1][j] + iimg[i+1][j+1]))
        mean = sum / 9
        new_img[i-1][j-1] = mean
plt.subplot(334), plt.imshow(new_img, 'gray'), plt.title('byh')
plt.axis('off')
new_img = np.array(new_img)
new_img = Image.fromarray(new_img)
new_img = new_img.filter(ImageFilter.MinFilter (size = 5))#最小滤波
new_img = np.array(new_img)
plt.subplot(335), plt.imshow(new_img, 'gray'), plt.title('zxlv')
plt.axis('off')
ret, new_img = cv.threshold(new_img,10,255,cv.THRESH_BINARY)
plt.subplot(336), plt.imshow(new_img, 'gray'), plt.title('yzcl')
plt.axis('off')
new_img  = cv.medianBlur(new_img, 3)
plt.subplot(337), plt.imshow(new_img, 'gray'), plt.title('zzlb')
plt.axis('off')
plt.show()