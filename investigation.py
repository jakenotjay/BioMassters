from utils import getFilesForChip, getRasters
import matplotlib.pyplot as plt

# chip id for the first chip in the training set
chip = "0003d2eb"

s1_files = getFilesForChip(chip, "S1")
s2_files = getFilesForChip(chip, "S2")
agb_file = getFilesForChip(chip, "AGB")[0] # only one file

# use xarray and rasterio to open the files
import xarray as xr
import rioxarray as rxr

agb = rxr.open_rasterio(agb_file)
print(agb)
print(agb.shape)

s1 = getRasters(s1_files)
# get the mean of the s1 data along the time dimension
s1 = s1.mean(dim="time")

s2 = getRasters(s2_files)
# get the mean of the s2 data along the time dimension
s2 = s2.mean(dim="time")

# create NDVI from s2
ndvi = (s2.sel(band=8) - s2.sel(band=4)) / (s2.sel(band=8) + s2.sel(band=4))

# squeze agb as it only has a single band
agb = agb.squeeze()

# combine ndvi and agb into a single xarray
agb = agb.rename("agb")
ndvi = ndvi.rename("ndvi")
combined = xr.merge([agb, ndvi])

# filter out for values where agb is zero
combined = combined.where(combined.agb > 0)

# plot the data
combined.plot.scatter(x="ndvi", y="agb")
plt.show()

# flatten x and y of combined for both ndvi and agb
# x = combined.ndvi.values.flatten()
# y = combined.agb.values.flatten()
# print(x.shape, y.shape)
# calc rcoeff
# import numpy as np
