# Importing files and scripts
from src import *
from clean_data import run_clean_functions
from feature_extraction import run_extract_features
from load_data import run_load_data
from tokenize import run_tokenize
from model import model

def main():
    
    # Clean the data using the clean_data script
    run_clean_functions()
    
    # Extracts the features vectors and saves them as a h5 file called features
    run_extract_features()
    
    # Loads cleaned captions for image image in a dictionary and stores the feature vectors for each image
    descriptions, features = run_load_data()
    
    # Tokenizes each word in the captions vocab and stores it in a pickle file called tokenizer
    tk, MAX_LENGTH, VOCAB_SIZE = run_tokenize(descriptions)
    
    # Initiates model, trains, and tests it.
    model(epochs=5, 
          descriptions=descriptions, 
          features=features, 
          tokenizer=tk, 
          max_length=MAX_LENGTH, 
          vocab_size=VOCAB_SIZE)
    

if __name__ == "__main__":
    
    main()
