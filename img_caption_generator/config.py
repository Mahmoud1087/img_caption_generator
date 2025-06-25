from pathlib import Path
import string
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import tensorflow as tf
import pandas

# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

project_root = Path(__file__).resolve().parents[1]

data_raw_dir = Path(f"{project_root}/data/raw/Flickr8k_text/")
data_processed_dir = Path(f"{project_root}/data/processed")
data_cleaned_dir = Path(f"{project_root}/data/cleaned")