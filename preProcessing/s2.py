"""A module for processing sentinel2 data.

The following 11 bands are provided for each S2 image: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, and CLP (a cloud probability layer). 
"""
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
from typing import List

def scale(img: xr.DataArray, scaleFactor: float = 0.0001) -> xr.DataArray:
    """Multiply the image by scaleFactor

    Args:
        img (xr.DataArray): image to scale
        scaleFactor (float, optional): scale factor. Defaults to 0.0001.

    Returns:
        xr.DataArray: scaled image
    """
    return img * scaleFactor

def normalize(img: xr.DataArray) -> xr.DataArray:
    """Normalize the image to between 0 and 1

    Args:
        img (xr.DataArray): image to normalize

    Returns:
        xr.DataArray: normalized image
    """

    min = img.min()
    max = img.max()

    return (img - min) / (max - min)

def process(images: List[xr.DataArray], cldProb: int = 50, ) -> xr.DataArray:
    """Process a list of s2 images.
    
    Args:
        images (List[xr.DataArray]): list of images to process
        cldProb (int, optional): cloud probability threshold. Defaults to 50.

    Returns:
        xr.DataArray: processed image
    """

    # combine images to a single xr.DataArray
    img = xr.concat(images, dim="time")

    # select cldProb band
    cldImage = img.isel(band=10)

    # select rest of bands
    img = img.isel(band=slice(0, 10))

    # scale bands
    img = scale(img)

    # filter out cloudy pixels
    img = img.where(cldImage < cldProb)

    # create median composite
    img = img.median(dim="time")

    # return image
    return img

def plot(img: xr.DataArray):
    """Clip, normalize and plot the image
    
    Args:
        img (xr.DataArray): image to plot
    """
    thresholded = img.clip(min = 0, max = 0.3)
    normalized = normalize(thresholded)

    plt.figure()
    normalized.plot.imshow()
    plt.show()