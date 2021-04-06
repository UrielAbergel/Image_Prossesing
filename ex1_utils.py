"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as img

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315660720


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # conver the img
    if representation == LOAD_RGB:
        image = cv.imread(filename, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    elif representation == LOAD_GRAY_SCALE:
        image = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    else:
        print("cant convert image to representation")

    return image / 255.0

    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    imag = imReadAndConvert(filename, representation)

    if representation == LOAD_GRAY_SCALE:
        plt.imshow(imag, cmap='gray')
    else:
        plt.imshow(imag)

    plt.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    matYIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.532, 0.311]])
    return np.reshape((np.dot(imgRGB.reshape(-1, 3), matYIQ.transpose())), imgRGB.shape)
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    matYIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.532, 0.311]])
    return np.reshape(np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(matYIQ).transpose()), imgYIQ.shape)

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # ============ check if the color is RGB or Gray
    color = checkIfColor(imgOrig)
    if color:
        n, m, k = imgOrig.shape
    else:
        n, m = imgOrig.shape

    # convert to YIQ to take the Y
    if color:
        # ============ take the Y array to perform
        image = transformRGB2YIQ(imgOrig)[:, :, 0]
    else:
        image = imgOrig.copy()

    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    image = image.astype('uint8')

    # ============ make histogram
    histOrg, old_bins = np.histogram(image.ravel(), 256)
    # ============ calculate the cumsum
    cumsum = np.cumsum(histOrg)
    # ============   create lookuptable
    cumsum = cumsum / np.max(cumsum)
    lookUpTable = cumsum * 255
    lookUpTable = lookUpTable.astype('uint8')
    # enter all parameter to the look up table
    imEq = lookUpTable[image.ravel()]
    if not color:
        histEQ, new_bins = np.histogram(imEq.ravel(), 256)
        return imEq.reshape(n, m), histOrg, histEQ
    else:
        histEQ, new_bins = np.histogram(imEq.ravel(), 256)
        newImage = transformRGB2YIQ(imgOrig)
        newImage[:, :, 0] = imEq.reshape(n, m) / 255
        newImage = transformYIQ2RGB(newImage)
        return newImage, histOrg, histEQ


def checkIfColor(img: np.ndarray) -> bool:
    dimOfMath = img.ndim
    # check if is gray scale of rgb
    if dimOfMath == 3:
        imgIsColor = True
    else:
        imgIsColor = False
    return imgIsColor


def goodValue(img: np.ndarray, row, cols):
    return img[row][cols] < 255


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    zArray = []
    qArray = []
    errArray = []
    picsArray = []

    color = checkIfColor(imOrig)
    # convert to YIQ to take the Y
    if color:
        # ============ take the Y array to perform
        image = transformRGB2YIQ(imOrig)[:, :, 0]
    else:
        image = imOrig.copy()

    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    hist, bins = np.histogram(image, 256)
    border = int(256 / nQuant)
    for i in range(nQuant + 1):
        zArray.append(i * border)
    for i in range(nIter):
        qArray = findMyQarray(nQuant, hist, zArray)
        errArray.append(returnMyError())




def findMyQarray(nQuant: int, hist: np.ndarray, zArray: np.ndarray) -> np.ndarray:
    cumsum = cumsum = np.cumsum(hist)
    zArrayQtoReturn = []
    sumOfall = 0
    numbers = 0

    for i in range(nQuant):
        getStart = zArray[i]
        getEnd = zArray[i+1]
        for j in range(getStart, getEnd):
            sumOfall = sumOfall + hist[j] * j
            numbers = numbers + hist[j]
        zArrayQtoReturn.append(sumOfall/numbers)
        sumOfall = 0
        numbers = 0

    return zArrayQtoReturn

def returnMyError() -> np.ndarray:
    print("hello")


def sumArrayFromAtoB(hist: np.ndarray,a : int, b: int) -> int:
    sum = 0
    for i in range(b-a):
        sum = sum + hist[i]
    return sum


if __name__ == '__main__':
    imag = imReadAndConvert('beach.jpg', LOAD_RGB)
    # pltooo = transformRGB2YIQ(imag)
    # plt.imshow(pltooo)
    # plt.show()
    # pltooo = transformYIQ2RGB(pltooo)
    # plt.imshow(pltooo)
    # plt.show()
    # imDisplay("beach.jpg", LOAD_RGB)

    hsitogramEqualize(imag)
