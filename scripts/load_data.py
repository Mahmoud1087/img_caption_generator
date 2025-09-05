from src import load_images, load_cleaned_images, load_features, text_data_raw_dir, data_cleaned_dir

IMG_TEXT_FILES_PATH = text_data_raw_dir / "Flickr_8k.trainImages.txt"
CLEANED_IMG_CAPTIONS = data_cleaned_dir / "Flickr8k.token_cleaned.txt"


def run_load_data():
    
    # Load image file names into a list
    images = load_images(filename=IMG_TEXT_FILES_PATH)

    # Load only the image captions dictionary of the images in the images list
    descriptions = load_cleaned_images(filename=CLEANED_IMG_CAPTIONS, images=images)

    # Load feature vectors of the images in the images list
    features = load_features(images=images, vectors="../models/features.h5")
    
    # Reshaping the vector to (2048,) instead of (1, 10, 10, 2048)        
    features = {img:np.mean(features[img][0], axis=(0, 1)) for img in images} 
    
    return descriptions, features

if __name__ == "__main__":
    
    run_load_data()