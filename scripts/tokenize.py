from src import create_tokenizer, max_length

def run_tokenize(descriptions:dict):

    # Tokenize the vocabulary of the image captions
    tokenizer = create_tokenizer(img_captions=descriptions)
    
    # Returns the length of the longest caption
    MAX_LENGTH = max_length(img_captions=descriptions)
    
    VOCAB_SIZE = len(tk.word_index) + 1
    
    return tokenizer, MAX_LENGTH, VOCAB_SIZE

if __name__ == "__main__":
    
    run_tokenize(descriptions)