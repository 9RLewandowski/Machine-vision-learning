import time

import cv2
import numpy as np

image_path = "D:\pythonProject\OpenCVProject\Test\sobel_test.jpg"  # 图片路径

# 读取显示图片
# image_path = "D:\pythonProject\OpenCVProject\Test\scores.jpg"  # 图片路径
# img = cv2.imread(image_path)
# cv2.imshow('img_window', img)  # 显示图片,[图片窗口名字，图片]
# cv2.waitKey(0)  # 无限期显示窗口
# cv2.destroyAllWindows()

# 将图片的三个通道分开
# b, g, r = cv2.split(img)
# cv2.imshow("Blue_1", b)
# cv2.imshow("Green_1", g)
# cv2.imshow("Red_1", r)
# cv2.waitKey(0)  # 无限期显示窗口
# cv2.imwrite("D:\pythonProject\OpenCVProject\Test\scores1.jpg", b)


# OpenCV
ori_img = cv2.imread(image_path)
x = cv2.Sobel(ori_img[:, :, 0], cv2.CV_16S, 1, 0)
y = cv2.Sobel(ori_img[:, :, 0], cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


# my sobel
def convertu8(num):
    if num > 255 or num < -255:
        return 255
    elif -255 <= num <= 255:
        if abs(num - int(num)) < 0.5:
            return np.uint8(abs(num))
        else:
            return np.uint8(abs(num)) + 1


def sobel(img, k=0):
    row = img.shape[0]
    col = img.shape[1]
    image = np.zeros((row, col), np.uint8)
    s = time.time()

    #Gy# -1 0 1
    ### -1 0 2
    ### -1 0 1
    # Gx# -1 -1 -1
    ###    0 0 0
    ###    1 2 1
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            y = int(img[i - 1, j + 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i, j + 1, k]) - int(img[i, j - 1, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i + 1, j - 1, k])
            x = int(img[i + 1, j - 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i + 1, j, k]) - int(img[i - 1, j, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i - 1, j + 1, k])
            image[i, j] = convertu8(abs(x) * 0.5 + abs(y) * 0.5)

    #Gx# -1 0 1
    ### -2 0 2
    ### -1 0 1
    #Gy#  1 2 1
    ###   0 0 0
    ###  -1 -2 -1
    # for i in range(1, row - 1):
    #     for j in range(1, col - 1):
    #         y = int(img[i - 1, j + 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
    #                 int(img[i, j + 1, k]) - 2*int(img[i, j - 1, k])) + int(img[i + 1, j + 1, k]) - int(
    #             img[i + 1, j - 1, k])
    #         x = int(img[i - 1, j - 1, k]) - int(img[i + 1, j - 1, k]) + 2 * (
    #                 int(img[i - 1, j, k]) - 2 * int(img[i + 1, j, k])) - int(img[i + 1, j + 1, k]) + int(
    #             img[i - 1, j + 1, k])
    #         image[i, j] = convertu8(abs(x) * 0.5 + abs(y) * 0.5)
    e = time.time()
    print(e - s)
    return image


sobelimage = sobel(ori_img, 0)
cv2.imshow('origin_image', ori_img)
cv2.imshow("OpenCV's Result", dst)
cv2.imshow("myresult", sobelimage)

cv2.waitKey(0)
cv2.destroyAllWindows()
