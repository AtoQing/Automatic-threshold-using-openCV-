import cv2
import numpy as np

img=cv2.imread("image.jpg")
img = cv2.resize(img,(500,500))
cv2.imshow("Original image",img)

image_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([image_gray],[0],None,[255],[0,255])


within=[]
for i in range(len(histogram)):
    pixel_sayisi = img.shape[0]*img.shape[1]
    background,foreground = np.split(histogram, [i])
    weight_background = np.sum(background)/pixel_sayisi
    weight_foreground = np.sum(foreground)/pixel_sayisi

    mean_background = np.sum([j * t for j, t in enumerate(background)]) / np.sum(background)
    mean_foreground = np.sum([j * t for j, t in enumerate(foreground)]) / np.sum(foreground)

    variance_background = np.sum([(j - mean_background) ** 2 * t for j, t in enumerate(background)]) / np.sum(background)
    variance_background = np.nan_to_num(variance_background)
    variance_foreground = np.sum([(j - mean_foreground) ** 2 * t for j, t in enumerate(foreground)]) / np.sum(foreground)
    within.append(weight_background*variance_background+weight_foreground*variance_foreground)

min = np.argmin(within)

(tresh,Bin)=cv2.threshold(image_gray,min,255,cv2.THRESH_BINARY)


cv2.imshow("Binary image",Bin)


cv2.waitKey(0)
cv2.destroyAllWindows()