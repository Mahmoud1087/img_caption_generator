# Importing files and scripts
from src import *
from clean_data import run_clean_functions
from feature_extraction import run_extract_features

def main():
    
    # Clean the data using the clean_data script
    run_clean_functions()
    
    # Extracts the features vectors and saves them as a pickle file called features
    run_extract_features()
    

if __name__ == "__main__":
    
    main()
