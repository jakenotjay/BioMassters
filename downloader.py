"""
Downloader for downloading biomass data from S3.

The data folders for the competition are hosted on a public AWS S3 bucket. Within the bucket, there are separate folders for the training set features ("train_features", which contains satellite data), testing set features ("test_features"), and training set labels ("train_agbm", which contains LiDAR ground-truth data).

The following directory structure is used:

|-- features_metadata.csv
|-- train_features
|   |__<satellite files>
|-- test_features
|   |__ <satellite files>
|-- train_agbm_metadata.csv
|-- train_agbm
    |__ <LiDAR files>

Data folders can be downloaded from the following links:

    - training set features: s3://drivendata-competition-biomassters-public-eu/train_features/
    - test set features: s3://drivendata-competition-biomassters-public-eu/test_features/
    - training set AGBM: s3://drivendata-competition-biomassters-public-eu/train_agbm

The satellite feature data files are named `{chip_id}_{satellite}_{month}.tif`, where `month` represents the number of months starting from September (00 is September, 01 is October, 02 is November, and so on). The LiDAR AGBM files are named `{chip_id}_agbm.tif`.


## Satellite metadata

We have also provided a "features_metadata.csv" file on the "Data Download" page that contains metadata about hosted satellite data. This may be useful if you want to write a script for downloading. The metadata file also includes file hashes that can be used to verify the integrity of a downloaded file. Hashes are generated using the default cksum hash function.

"features_metadata.csv" contains the following columns:

- `chip_id`: A unique identifier for a single patch, or area of forest
-* `filename`: The filename of the corresponding image, which follows the naming convention `{chip_id}_{satellite}_{month_number}.tif`. (`month_number` corresponds to the number of months since September of the year previous to when the ground truth was captured, so `00` would represent September, `01` October, and so on, until `12`, which represents August of the same year)
- `satellite`: The satellite the image was captured by (`S1` Sentinel-1 or `S2` for Sentinel-2)
- `split`: Whether the image is a part of the training data or test data
- `month`: The name of the month in which the image was collected
- `size`: The file size in bytes
- `cksum`: A checksum value to make sure the data was transmitted correctly
- `s3path_us`: The file location of the image in the public s3 bucket in the US East (N. Virginia) region
- `s3path_eu`: The file location of the image in the public s3 bucket in the Europe (Frankfurt) region
- `s3path_as`: The file location of the image in the public s3 bucket in the Asia Pacific (Singapore)
- `corresponding_agbm`: The filename of the tif that contains the AGBM values for the chip_id

To check that your data was not corrupted during download, you can generate your own hashes at the command line and compare them to the metadata. For example, we know from the metadata that the hash for the file "001b0634_S1_00.tif" is 3250666344 and the byte count is 1049524. To generate a checksum value for a locally saved version:

	$ cksum test/001b0634_S1_00.tif
	3250666344 1049524 test/001b0634_S1_00.tif

## LiDAR metadata

We have also provided a "train_agbm_metadata.csv" file on the "Data Download" page that contains metadata about the ground-truth AGBM measures acquired using LiDAR. This may be useful if you want to write a script for downloading. Like "features_metadata.csv", "train_agbm_metadata.csv" also includes file hashes  generated using the default cksum hash function that can be used to verify the integrity of a downloaded file.

"train_agbm_metadata.csv" contains the following columns:

- `chip_id`: The patch the image corresponds to
- `filename`: The filename the image can be found under. The filename follows the convention `{chip_id}_agbm.tif`
- `size`: The file size in bytes
- `cksum`: A checksum value to make sure the data was transmitted correctly
- `s3path_us`: The file location of the image in the public s3 bucket in the US East (N. Virginia) region
- `s3path_eu`: The file location of the image in the public s3 bucket in the Europe (Frankfurt) region
- `s3path_as`: The file location of the image in the public s3 bucket in the Asia Pacific (Singapore)

"""

import boto3
import os
import pandas as pd

from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="eu-central-1")



def downloadFile(row, download_dir, region, folder):
    """Download a single file.

    Args:
        row (pandas.Series): The row of metadata for the file to download.
        download_dir (str): The directory to download the file to.
        region (str): The AWS region to download the file from.

    """
    filename = row.filename
    s3path = row[f"s3path_{region}"]

    bucket = s3path.split("//")[1].split("/")[0]
    s3path = "/".join(s3path.split("//")[1].split("/")[1:])

    filepath = os.path.join(download_dir, folder, filename)
    print("filepath", filepath)
    if not os.path.exists(filepath):
        # print bucket and s3path
        print("bucket", bucket)
        print("s3path", s3path)
        print(f"Downloading {filename} to {download_dir}...")
        s3.download_file(bucket, s3path, filepath)
    else:
        print(f"File {filename} already exists in {download_dir}.")

def downloadFilesForChipId(chip_id, download_dir, metadata, region, folder):
    """Download all files for a given chip_id.

    Args:
        chip_id (str): The chip_id to download files for.
        download_dir (str): The directory to download files to.
        metadata (pandas.DataFrame): The metadata for the files to download.
        region (str): The AWS region to download files from.

    """
    for _, row in metadata[metadata.chip_id == chip_id].iterrows():
        downloadFile(row, download_dir, region, folder)

def checkForDownloadDir(download_dir):
    """Checks if the download directory exists, and creates it if it doesn't.

    Args:
        download_dir (str): The directory to check.

    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(os.path.join(download_dir, "train_features")):
        os.makedirs(os.path.join(download_dir, "train_features"))
    if not os.path.exists(os.path.join(download_dir, "train_agbm")):
        os.makedirs(os.path.join(download_dir, "train_agbm"))
    
def downloadNChips(n, download_dir, region, metadataFolder="meta"):
    """Downloads N random files from the dataset, including both satellite and Lidar
    
    Args:
        n (int): The number of chips to download.
        download_dir (str): The directory to download files to.
        region (str): The AWS region to download files from.
    """
    checkForDownloadDir(download_dir)

    # join with meta folder
    path = metadataFolder

    # open agbm metadata
    agbm_metadata = pd.read_csv(os.path.join(path, "train_agbm_metadata.csv"))
    # open features metadata
    features_metadata = pd.read_csv(os.path.join(path, "features_metadata.csv"))

    # choose a random subset of the agbm metadata
    agbm_metadata = agbm_metadata.sample(n=n)

    # loop over the row in agbm metadat
    for _, row in agbm_metadata.iterrows():
        chip_id = row.chip_id
        # download the corresponding satellite images
        downloadFilesForChipId(chip_id, download_dir, features_metadata, region, "train_features")
        # download the corresponding agbm file
        downloadFile(row, download_dir, region, "train_agbm")