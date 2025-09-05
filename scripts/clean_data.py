from src import img_captions, simplify, save_captions, data_raw_dir, data_cleaned_dir


TOKEN_PATH = data_raw_dir / "Flickr8k.token.txt"

def run_clean_functions():
    #  Creates dictionary of each image and its captions
    img_caps = img_captions(TOKEN_PATH)

    # Removes punctuations, words that are less than 2 characters long and numbers
    img_caps_simplified = simplify(img_caps)

    # Saves new cleaned file in the same format as the original file in the cleaned directory
    save_captions(img_caps_simplified, data_cleaned_dir / "Flickr8k.token_cleaned.txt")


if __name__ == "__main__":
    
    run_clean_functions()
    
    