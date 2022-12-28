"""Module for pre-processing the data."""

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import xarray as xr
import rioxarray as rxr
from typing import List

from .s2 import process as processS2
from .s2 import plot as plotS2

def openFiles(files: List[str]) -> List[xr.DataArray]:
    """Open a list of files and return a list of xr.DataArray

    Args:
        files (List[str]): list of file paths

    Returns:
        List[xr.DataArray]: list of xr.DataArray
    """
    return [rxr.open_rasterio(f) for f in files]

def preProcess(files: List[str]) -> xr.DataArray:
    """Pre-process a list of files and return a single xr.DataArray

    Args:
        files (List[str]): list of file paths

    Returns:
        xr.DataArray: single xr.DataArray
    """
    s2Rasters = [f for f in files if "S2" in f]

    s2Rasters = openFiles(s2Rasters)

    s2 = processS2(s2Rasters)

    plotS2(s2)

    return s2