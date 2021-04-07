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
from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import imReadAndConvert
import cv2 as cv
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
title = 'gamma correction'




def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == LOAD_RGB:
        image = cv.imread(img_path, cv.IMREAD_COLOR)

    elif rep == LOAD_GRAY_SCALE:
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    cv.namedWindow(title)

    def on_trackbar(val):
        gamma = float(val)/100.0
        print(gamma)
        Corrected = np.power((image.copy()/ 255), gamma) * 255
        Corrected = Corrected.astype('uint8')
        print(Corrected)
        cv.imshow(title, Corrected)

    trackbar_name = 'gama x %d' % 2
    cv.createTrackbar(trackbar_name, title, 0, 200, on_trackbar)






    on_trackbar(0)
    # Wait until user press some key
    cv.waitKey()
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
