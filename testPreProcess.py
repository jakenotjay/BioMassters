import os

# set the environment variables
os.environ['AGB_DATA'] = "./data/train_agbm"
os.environ['TRAIN_DATA'] = "./data/train_features"

from utils import getFilesForChip
from preProcessing.preProcess import preProcess

chip = "288afc34"

files = getFilesForChip(chip)

s2 = preProcess(files)

# print min and max
print(s2.min().values, s2.max().values)
