import numpy as np

def rgb2gray(rgb):
    """Converts image to grayscale

    Args:
        State (np.array): This is a RGB color array from pygames.

    Returns:
        ndarray of unit8: grayscale and reduced image of pygames screen

    """
    rgb = rgb/255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    # The standard luminance calculation for RGB -> grayscale.
    s_gray = (r*.2126 + g*.7152 + b*.0722)*255

    return s_gray.astype(np.uint8)