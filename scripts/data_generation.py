from src import data_generator

def run_data_generator(descriptions, image_feature_vectors, tk, max_caps, vocab_size):
    
    data_generator(descriptions=descriptions, 
                   features=image_feature_vectors, 
                   tokenizer= tk, 
                   max_length=max_caps, 
                   vocab_size=vocab_size)
    
if __name__ == "__main__":
    
    run_data_generator()