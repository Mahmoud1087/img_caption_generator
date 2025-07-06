from src import extract_features, img_data_raw_dir
import h5py

IMG_PATH = img_data_raw_dir

def run_extract_features():
    
    # Extract feature vector for each image and return a dictionary of each image and its vector
    features = extract_features(IMG_PATH)

    # Save the dictionary into a pickle file
    with h5py.File('../models/features.h5', 'w') as f:
        for k, v in features.items():
            f.create_dataset(k, data=np.array(v), compression="gzip")
    

if __name__ == "__main__":
    
    run_extract_features()