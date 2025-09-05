from pathlib import Path



project_root = Path(__file__).resolve().parents[1]

text_data_raw_dir = Path(f"{project_root}/data/raw/Flickr8k_text/")
data_processed_dir = Path(f"{project_root}/data/processed")
data_cleaned_dir = Path(f"{project_root}/data/cleaned")

img_data_raw_dir = Path(f"{project_root}/data/raw/Flickr8k_Dataset/Flicker8k_Dataset")

