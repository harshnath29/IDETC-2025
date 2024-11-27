### TESTER FOR FINAL PROJECT

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

# Reading in the Image
src = cv.imread('WIN_20241121_17_58_56_Pro.jpg')
cv.namedWindow("Source", cv.WINDOW_NORMAL)
cv.imshow('Source', src) # Show image
cv.waitKey() # Wait to exit

# Convert to HSV
img_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

## SHIFT HUE SPACE to decrease fuzziness on edges
shift = 90
img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(int) + shift) % 180

cv.namedWindow("Source", cv.WINDOW_NORMAL)
cv.imshow('Source',img_hsv)
cv.waitKey()

# Bounds for color selection
lower_bound = np.array([0,90,0])
upper_bound = np.array([255,255,255])

mask = cv.inRange(img_hsv, lower_bound, upper_bound)
# invert_mask = cv.bitwise_not(mask)

img_iso = cv.bitwise_and(src, src, mask=mask)

cv.namedWindow("image mask", cv.WINDOW_NORMAL)
cv.imshow('image mask', mask)
cv.namedWindow("isolated image", cv.WINDOW_NORMAL)
cv.imshow('isolated image', img_iso)
cv.waitKey()

## Canny edge detection
threshold1 = 90
threshold2 = 180

# Dilation
kernel = np.ones((15,15), np.uint8)
dilation = cv.dilate(mask,kernel,iterations = 1)

src_processed = cv.blur(dilation, (3,3))
edges = cv.Canny(src_processed, threshold1, threshold2)

plt.subplot(121), plt.imshow(src) # Note: colors swapped b/c OpenCV stores colors in BGR, but matplotlib does in RGB
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges)
plt.title('Edges Image'), plt.xticks([]), plt.yticks([])
plt.show()

## CONTOURS
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    cv.drawContours(drawing, contours, i, (255, 255, 255))

cv.namedWindow("Contours", cv.WINDOW_NORMAL)
cv.imshow('Contours', drawing)
cv.waitKey()

x = list()
y = list()
for i in contours:
    for j in i:
        row, col = j[0]
        if col > 2000:
            x.append(row)
            y.append(-1 * col)


plt.scatter(x, y, s = 1)
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.xlim(0, 2500)
plt.ylim(-3000, 0)
plt.show()


### Add boundingRect
oneContour = contours[0]
for i in range(1, len(contours)):
    oneContour = np.vstack((oneContour, contours[i]))
oneRect = cv.boundingRect(oneContour)
print(oneRect)

cv.rectangle(src, (oneRect[0], oneRect[1]), \
    (oneRect[0]+ oneRect[2], oneRect[1]+ oneRect[3]), (255, 255, 255), 2)
cv.namedWindow("boundRect", cv.WINDOW_NORMAL)
cv.imshow('boundRect', src)
cv.waitKey()
x1min = min(x)
x1max = max(x)


### DRAW WITH APPROXPOLYDP to close curves


### Approximate difference w/ HausdorffDistanceExtractor? or ShapeContextDistanceExtractor or matchShapes
src = cv.imread('IMGB.jpg')
img_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
shift = 90
img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(int) + shift) % 180
lower_bound = np.array([0,90,0])
upper_bound = np.array([255,255,255])
mask = cv.inRange(img_hsv, lower_bound, upper_bound)
img_iso = cv.bitwise_and(src, src, mask=mask)
threshold1 = 90
threshold2 = 180
kernel = np.ones((15,15), np.uint8)
dilation = cv.dilate(mask,kernel,iterations = 1)
src_processed = cv.blur(dilation, (3,3))
edges = cv.Canny(src_processed, threshold1, threshold2)
plt.subplot(121), plt.imshow(src) # Note: colors swapped b/c OpenCV stores colors in BGR, but matplotlib does in RGB
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges)
plt.title('Edges Image'), plt.xticks([]), plt.yticks([])
plt.show()
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    cv.drawContours(drawing, contours, i, (255, 255, 255))
cv.namedWindow("Contours", cv.WINDOW_NORMAL)
cv.imshow('Contours', drawing)
cv.waitKey()
x = list()
y = list()
for i in contours:
    for j in i:
        row, col = j[0]
        if col > 2000:
            x.append(row)
            y.append(-1 * col)
plt.scatter(x, y, s = 1)
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.xlim(0, 2500)
plt.ylim(-3000, 0)
plt.show()

twoContour = contours[0]
for i in range(1, len(contours)):
    twoContour = np.vstack((twoContour, contours[i]))
twoRect = cv.boundingRect(twoContour)
print(twoRect)

cv.rectangle(src, (twoRect[0], twoRect[1]), \
    (twoRect[0]+ twoRect[2], twoRect[1]+ twoRect[3]), (255, 255, 255), 2)
cv.namedWindow("boundRect", cv.WINDOW_NORMAL)
cv.imshow('boundRect', src)
cv.waitKey()

print(cv.matchShapes(oneContour,twoContour,3,0))
x2min = min(x)
x2max = max(x)

print(x1min - x2min)
print(x1max - x2max)


## Truncate based on a percentage of height OR based on the first picture. I.e. store the layer height from the first picture
# then use the y-value from that picture as where to truncuate the next photo and so on
