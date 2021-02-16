
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


def convolve_greyscale(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros((int((image.shape[0] - kernel.shape[0] + 2) + 1),
                       int((image.shape[1] - kernel.shape[1] + 2) + 1)))
    img_pad = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    img_pad[1:-1, 1:-1] = image

    for y in range(img_pad.shape[1]):
        if y > img_pad.shape[1] - kernel.shape[1]:
            break
        for x in range(img_pad.shape[0]):
            if x > img_pad.shape[0] - kernel.shape[0]:
                break

            output[x, y] = (kernel * img_pad[x: x + kernel.shape[0],
                                     y: y + kernel.shape[1]]).sum()

    return output


def pooling(image, kernel, stride):
    img_tmp = image
    x, y = image.shape[:2]
    kernel_y, kernel_x = kernel
    stride_y, stride_x = stride

    image = img_tmp[
            : math.floor((x - kernel_y) / stride_y * stride_y) + kernel_y,
            : math.floor((y - kernel_x) / stride_x * stride_x) + kernel_x]

    stride_A, stride_B = image.strides[:2]
    x_A, y_A = image.shape[:2]
    kernel_A, kernel_B = kernel
    result_tmp = (math.floor(1 + (x_A - kernel_A) / stride_y),
                  math.floor(1 + (y_A - kernel_B) / stride_x), kernel_A,
                  kernel_B) + image.shape[2:]
    strides = (stride_y * stride_A, stride_x * stride_B, stride_A,
               stride_B) + image.strides[2:]
    result = np.lib.stride_tricks.as_strided(image, result_tmp,
                                             strides=strides)

    return result


def max_pooling(image, kernel, stride=None):
    view = pooling(image, kernel, stride)
    result = np.nanmax(view, axis=(2, 3))

    return result


def average_pooling(image, kernel, stride):
    view = pooling(image, kernel, stride)
    result = np.nanmean(view, axis=(2, 3))

    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
