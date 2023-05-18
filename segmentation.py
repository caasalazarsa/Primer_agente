import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.filters import sobel
import numpy as np
import PIL.Image as Image
import libtiff
import cv2

im = Image.open('a.tif')
# IOError: cannot identify image file

tif = libtiff.TIFF.open('a.tif')
im = tif.read_image()
# im only contains one of the three channels. im.dtype is uint16 as desired.
im = []
for i in tif.iter_images():
    # still only returns one channel

im = np.array(cv2.imread('a.tif'))


#ximg = data.coffee()
img = cv2.imread(sys.argv[1],-1)

inputname=str(sys.argv[1])
outputname_cn=inputname.replace(".tiff", "seg.tiff")

outputname_cn=outputname_cn.replace("INPUT", "OUTPUT_CN")
#img = rgb2gray(img

denoised = cv2.GaussianBlur(img,(3,3),0)
filtered = cv2.Laplacian(denoised,cv2.CV_64F)

filtered2 = sobel(denoised)

#plt.imshow(filtered,cmap='gray')
#plt.show()
#plt.imshow(filtered2,cmap='gray')
#plt.show()
#cv2.imshow('filtered',filtered)
#cv2.waitKey(0)

res=np.uint8(0.5*img+2*filtered+0*filtered2)

#plt.imshow(res,cmap='gray')
#plt.show()

w,h = img.shape

s = np.linspace(0, 2*np.pi,360)
x = h/2 + (h/2)*np.cos(s)
y = w/2 + (w/2)*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(gaussian(res, 3),
                       init, alpha=0.00618, beta=5, gamma=0.0001)
    
#snake = active_contour(res,
 #                      init, alpha=0.00618, beta=5, gamma=0.0001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
#ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.savefig(outputname_cn)

