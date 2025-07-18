from src import create_tokenizer, max_length

def run_tokenize(descriptions:dict):

    # Tokenize the vocabulary of the image captions
    create_tokenizer(img_captions=descriptions)
    
    # Returns the length of the longest caption
    MAX_LENGTH = max_length(img_captions=descriptions)
    
    return MAX_LENGTH

if __name__ == "__main__":
    
    run_tokenize(descriptions)