from typing import Optional, List
import os

import xarray as xr
import rioxarray as rxr

AGB_FOLDER = os.getenv('AGB_DATA', "./data/agb")
TRAIN_FOLDER = os.getenv('TRAIN_DATA', "./data/train")

platformToFiles = {
    'AGB': lambda chip: [os.path.join(AGB_FOLDER, f) for f in os.listdir(AGB_FOLDER) if f.startswith(chip)],
    'S1': lambda chip: [os.path.join(TRAIN_FOLDER, f) for f in os.listdir(TRAIN_FOLDER) if f.startswith(chip) and "S1" in f],
    'S2': lambda chip: [os.path.join(TRAIN_FOLDER, f) for f in os.listdir(TRAIN_FOLDER) if f.startswith(chip) and "S2" in f]
}

def getFilesForChip(chip: str, platform: Optional[str] = None) -> List[str]:
    """For a given chip return the files. If no platform is given returns all files for the chip in agb, s1, and s2 order.
    
    Args:
        chip (str): chip id
        platform (Optional[str], optional): platform to get files for. Defaults to None.
    
    Returns:
        List: list of file paths
    """
    if platform is None:
        return platformToFiles['AGB'](chip) + platformToFiles['S1'](chip) + platformToFiles['S2'](chip)

    if platform in platformToFiles:
        return platformToFiles[platform](chip)
    else:
        raise ValueError(f'Platform {platform} not found, must be one of {platformToFiles.keys()}')