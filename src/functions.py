import string
from PIL import Image
import keras
import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical

# Create function to load the files
def load_file(filename:str):
    """
    Takes a filename and loads its contents into a string.

    Args: 
        filename (string): path of the file that will be loaded.

    Returns:
        text (string): a string of all of the lines in the filename.
    """

    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    return text


# Create a function to seperate each image with its captions
def img_captions(filename:str):
    """
    Takes a file of images and all of its captions and creates a dictionary with each image as the key 
    and the values are all of its  captions.

    Args:
        filename (string): the name of the file with the images and captions pair.

    Returns:
        image_descriptions (dictionary): a dictionary with each image as the key and its captions in a list 
        as the value.
    """

    # loading the file
    file = load_file(filename)
    captions = file.split("\n")
    image_captions = {}

    for caption in captions[:-1]: # Last line is empty so iterating up to the line before the end
        img, caption = caption.split('\t') # Splitting the image and caption using the 'tab'
        if img[:-2] not in image_captions: # Last 2 in every image has the "#" and "caption number" 
            image_captions[img[:-2]] = [caption]
        else:
            image_captions[img[:-2]].append(caption)

    return image_captions


# Create a function to simplify captions
def simplify(captions:dict):
    """
    Takes a dictionary of image and its captions and removes punctuations, numbers, and changes uppercase 
    letters into lowercase.

    Args: 
        captions (dictionary): A dictionary of each image as the key and a list of captions as the value.

    Returns:
        captions (dictionary): Returns the same dictionary after simplifying the captions.
    """
    
    table = str.maketrans("", "", string.punctuation)
    
    for img, img_captions in captions.items():
        for i, caption in enumerate(img_captions):
            
            caption.replace("-"," ") # Replacing "-" with a blank space
            words = caption.split() # Splitting each word in the dictionary

            words = [word.lower() for word in words] 
            # Applying the translation table of removing the punctuation marks
            words = [word.translate(table) for word in words] 

            # Only keeping words that are more then 1 character long
            words = [word for word in words if len(word)>1] 
            words = [word for word in words if word.isalpha()] # Removing numbers

            caption = " ".join(words)
            captions[img][i] = caption

    return captions


# Create a function to make up the vocab used in the text
def make_vocab(captions:dict):
    """
    Takes a dictionary of image and its captions and finds all unique words used to make up a set of vocabulary.

    Args:
        captions (dictionary): A dictionary of each image as the key and a list of captions as the value.

    Returns:
        vocab (set): A set of unique words from all the captions.
    """

    vocab = set()

    for key in captions.keys():
        
        [vocab.update(c.split()) for c in captions[key]]

    return vocab    


# Create a function that saves the images and its new editted captions
def save_captions(captions:dict, filename:str):
    """
    Takes the new dictionary of image and its new editted captions and saves them 
    in the original formate of image - caption_number as a text file.

    Args:
        captions (dictionary): A dictionary of each image as the key and a list of captions as 
        the value.
        
        filename (string): The filename of the text that will be saved.
    """

    lines = []
    for img, image_captions in captions.items():
        for i, caption in enumerate(image_captions):
            line = img + "\t" + caption
            lines.append(line)

    text ="\n".join(lines)
    
    file = open(filename, "w")
    file.write(text)
    file.close()
    

# Create a function that extracts the feature vector from images in a directory
def extract_features(directory:str):
    """
    Takes in a path for the images directory and returns a dictionary of each image and its feature vector 
    that is extraced using the Xception model.

    Args:
        directory (string): A path of the directory where the raw image files are located.

    Returns:
        features (dictionary): A dictionary of each image and its feature vector.
    """
    
    # Don't want the final layers which output the classification prediction instead we stop at the raw 
    # features detected
    xception_feature_extraction_model = keras.applications.Xception(include_top=False, pooling='average') 
    features = {}
    
    directory = Path(directory)
    
    for img in os.listdir(directory):
        
        filename = directory / img
        
        image = Image.open(filename)
        image = image.resize((299, 299))
        # changing the shape of the image to (1, 299, 299, 3) as keras expect the input in batches even 
        # if batch of 1.
        image = np.expand_dims(image, axis=0) 
        image = image / 127.5
        image = image - 1.0
        
        feature = xception_feature_extraction_model.predict(image) # outputs the feature vector
        features[img] = feature
        
    return features


# Create a function that loads the image filenames into a list
def load_images(filename:str):
    """
    Takes in a path for a file and loads each line in a list.

    Args:
        filename (string): A path of the file where the image filenames are loaded.

    Returns:
        images (list): A list of all image filenames.
    """
    
    images = []
    with open(IMG_TEXT_FILES_PATH, 'r') as f:
        img_files = f.readlines()
        for file in img_files:
            images.append(file[:-1]) 
            
    return images


# Creates a function that loads the captions of each image in a dictionary
def load_cleaned_images(filename:str, images:list):
    """
    Takes in a path for a file and a list of image names and loads each image and its captions from the file into
    a dictionary.

    Args:
        filename (string): A path of the file where all the image filenames their captions are saved.
        
        images (list): A list of image filenames.

    Returns:
        descriptions (dictionary): A dictionary of all the images in the images list as keys and their captions 
        from the filename as their values.
    """
    
    with open(filename, 'r') as captions:
    
        # Load cleaned captions text file
        file = captions.readlines()
        descriptions = {}
        
        file = " ".join(file)

        for line in file.split("\n"):
            words = line.split() # Split each line into words --> ["img001.jpg", "man", "wears", ......]
            
            if len(words)<1:
                continue
            
            image, image_caption = words[0], words[1:]
            
            if image in images:
                # Since each image appears 5 times in the captions text file, we need to initiate it once in 
                # the descriptions dict with an empty list and then append to that list each new caption after 
                # adding the start and end identifiers
                if image not in descriptions:
                    descriptions[image] = []
                desc = "<start> " + " ".join(image_caption) + " <end>"
                descriptions[image].append(desc)
            
    return descriptions


# Create a function that loads the previously saved features only for the images in a list
def load_features(images:list, vectors:str):
    """
    Takes in a list of images and returns a dictionary of these images and their feature vectors

    Args:
        images (list): A list of image filenames.
        
        vectors (string): A path of the saved feature vectors.

    Returns:
        image_feature_vectors (dictionary): A dictionary of all the images in the images list as keys and their feature 
        vectors stored in the models directory.
    """
    
    features = {}
    with h5py.File(vectors, 'r') as f:
        for i, k in enumerate(f.keys()):
            features[k] = f[k][()]  # Load individual array
            
    # Reshaping the vector to (2048,) instead of (1, 10, 10, 2048)
    image_feature_vectors = {img:np.mean(features[img][0], axis=(0, 1)) for img in images} 
    
    return image_feature_vectors


# Create a function to convert the values in a dictionary into a list
def dict_to_list(img_Captions:dict):
    """
    Takes a dictionary and returns a list of all the values.
    
    Args:
        img_captions (dictionary): A dictionary of images as keys and their captions as keys.
        
    Returns:
        all_descs (list): A list of all the captions for each image in the dictionary.  
    """
    
    all_descs = []
    for img, captions in descriptions.items():
        [all_descs.append(caption) for caption in captions]
        
    return all_descs


# Create a function to tokenize the captions using the Tokenizer class
def create_tokenizer(img_captions:dict):
    """
    Takes a dictionary of images and their captions and instantiates a Tokenizer class to tokenize the 
    vocabulary in the captions.

    Args:
        img_captions (dict): A dictionary of images and their captions as values.
    """
    
    # Coverting the dictionary to a list using the dict_to_list function
    all_descs = dict_to_list(img_captions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_descs) # Fitting the tokenizer class to the list of captions
    
    return tokenizer


# Create a function to calculate the maximum length of the captions
def max_length(img_captions:dict):
    """
    Calculates the length of each caption in a dictionary and returns the maximum value.

    Args:
        img_captions (dict): A dictionary of images and their captions as values.
        
    Returns:
        max_len (int): The maximum length of a caption.
    """
    
    captions = dict_to_list(img_captions)
    
    length_of_caps = [len(caption.split()) for caption in captions]
    
    max_len = max(length_of_caps)
    
    return max_len


# Create a helper function to return an image, its feature vector, and its caption
def create_sequences(tokenizer, max_length:int, desc_list:list, feature:list, vocab_size:int):
    """
    Creates 3 arrays consisting of the feature vector of a given image, as well as a sequence of inputs, 
    and an output sequence that would be predicted as part of future training.

    Args:
        tokenizer (Tokenizer object): A tokenizer object created using keras.src.legacy.preprocessing.text.Tokenizer.
        max_length (int): The length of the longest caption in the captions available.
        desc_list (list): A list of all 5 captions for a specific image.
        feature (list): A feature vector of a specific image as a list.
        vocab_size (int): The vocabulary size of the dataset.

    Returns:
        [x1, x2], y: A set with the first element as a list consisting of the feature vector and the input 
                     sequence, while the second element is the output sequence.
    """
    # initialize empty lists for each parameter, where x1 is the input image, x2 is the input vector, and y is 
    # the output sequence
    x1, x2, y = list(), list(), list()
    
    # loop through each one of the 5 descriptions for each image
    for desc in desc_list:
        # encode the sequence into tokens using the tokenizer
        # Indexing on element 0 because the output is inside another vector
        seq = tokenizer.texts_to_sequences([desc])[0] 
        
        # split the sequence into multiple X:Y pairs
        for i in range(1, len(seq)):
            # split into input and output pairs in order to predict the next word based on the previous sequence
            in_seq, out_seq = seq[:i], seq[i]
            # pad the input sequence into the max legnth
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode the output sequence - becomes a vector of the size of the vocab size
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # This way the model can predict the next word based on the inupt image feature vector as well as 
            # the current input sequence
            x1.append(feature)
            x2.append(in_seq)
            y.append(out_seq)
            
    return np.array(x1), np.array(x2), np.array(y)


# Create a generator function that yields new input/output sequences for each image
def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    """
    A Python generator that yields batches of training data.

    Each iteration returns:
      - A tuple of model inputs: [input_img, input_vector]
          * input_img: NumPy array of image feature vectors (e.g., shape: (num_samples, feature_dim))
          * input_vector: NumPy array of tokenized and padded partial captions 
          (e.g., shape: (num_samples, max_length))
      - The corresponding target output (output_seq): one-hot encoded next-word vectors
          (e.g., shape: (num_samples, vocab_size)).

    Args:
        descriptions (dict): A dictionary of all the images in the images list as keys and their captions 
                             from the filename as their values.
        
        features (dict): A dictionary of each image and its feature vector.
        
        tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer fitted on the training captions, used 
                                                        to convert text to integer sequences.
        
        max_length (int): The maximum caption length for padding sequences.
        
        vocab_size (int): The total number of unique words in the vocabulary (used for one-hot encoding).

    Yields:
    
        tuple: ([[input_img, input_vector], output_seq]) for one image and its associated captions.
    """
    
    while True:
        for img, description_list in descriptions.items():
            feature = features[img]
            
            # Using the create_sequence function to generate the 3 arrays
            input_img, input_vector, output_seq = create_sequences(tokenizer=tk, 
                                                                   max_length=max_length, 
                                                                   desc_list=description_list, 
                                                                   feature=feature, 
                                                                   vocab_size=vocab_size)
            
            yield ( (input_img, input_vector), output_seq )
            

# Create a function to load images to test the model
def load_testing_imgs(directory:str, txt_file:str):
    """
    Selects 4 random images from a source directory, copies them into a directory,
    extracts features for each image, and reshapes each feature vector to (2048,).
    
    Args:
        directory (string): A path of the directory where the raw image files are located.
        
        txt_file (str): Path to a text file containing valid image filenames.

    Returns:
        features (dictionary): A dictionary of each image and its feature vector.
    """
    
    target_dir = Path("../data/raw/testing_images")
    
    # Making sure the path exists
    if not os.path.isdir(directory):
        raise ValueError(f"Source directory does not exist: {directory}")
    
    # Read image names from text file
    with open(txt_file, 'r') as f:
        all_filenames = [line.strip() for line in f if line.strip()]

    # Randomly select 4 from the list
    selected_images = random.sample(all_filenames, 4)

    # Create or clear target directory
    if os.path.exists(target_dir):
        for file in os.listdir(target_dir):
            os.remove(os.path.join(target_dir, file))
    else:
        os.makedirs(target_dir)

    # Copy selected images to target directory
    for img in selected_images:
        src = os.path.join(directory, img)
        dst = os.path.join(target_dir, img)
        shutil.copy2(src, dst)

    # Extract features for images in the target directory
    features = extract_features(target_dir)

    # Reshape to (2048,) if needed
    features = {key: np.mean(value[0], axis=(0, 1)) for key, value in features.items()}

    return features


# Create a function that uses the trained model to generate captions for each image
def generate_captions_for_images(model, tokenizer, features: dict, max_length: int):
    """
    Generate captions for multiple images using the trained model.

    Args:
        model (keras.Model): The trained image captioning model.
        
        tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer fitted on the training captions.
        
        features (dict): A dictionary of each image and its feature vector.
        
        max_length (int): Maximum caption length.

    Returns:
    
        results (dict): A dictionary mapping image filenames to their generated captions.
    """
    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
    
    results = {}
    
    for img_name, feature in features.items():
        
        in_text = 'start'
        for _ in range(max_length):
            
            seq = tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding='post')

            # Predict next word
            yhat = model.predict([np.expand_dims(feature, axis=0), seq], verbose=0)
            yhat = np.argmax(yhat)

            word = word_for_id(yhat, tokenizer)
            if word is None:
                break

            in_text += ' ' + word
            if word == 'end':
                break

        # Clean the caption
        caption = in_text.replace('start', '').replace('end', '').strip()
        results[img_name] = caption

    return results
        
# Create a function to display images and their captions 
def display_images_with_captions(captions_dict: dict, image_dir: str):
    """
    Displays images with captions in two rows.
    
    Args:
        captions_dict (dict): Mapping of image filenames to captions.
        image_dir (str): Directory containing the images.
    """
    num_images = len(captions_dict)
    cols = 2
    rows = (num_images + cols - 1) // cols  # ceiling division

    plt.figure(figsize=(12, 8))  # Adjust figure size

    for idx, (filename, caption) in enumerate(captions_dict.items()):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(caption, fontsize=10, wrap=True)

    plt.tight_layout()
    plt.show()
    
    
# Create a new function for generating captions using beam search instead of greedy search
def generate_captions_for_images_with_beam_search(model, tokenizer, features: dict, max_length: int, beam_index: int):
    
    """
        Generate captions for multiple images using the trained model.

        Args:
            model (keras.Model): The trained image captioning model.
            
            tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer fitted on the training captions.
            
            features (dict): A dictionary of each image and its feature vector.
            
            max_length (int): Maximum caption length.
            
            beam_index (int): Beam width (the number of parallel sequence paths to tepredictst before 
                            selecting the most probable).

        Returns:
        
            results (dict): A dictionary mapping image filenames to their generated captions.
        """
    def beam_search(model, tokenizer, features: list, max_length: int, beam_index: int):
        
        start = 'start'
        # Each element in `sequences` is a pair [caption_so_far, log_probability_score]
        sequences = [[start, 0.0]]

        while True:
            all_candidates = []

            # Expand each current candidate caption
            for caption, score in sequences:
                # If the caption already ends with 'end', we stop expanding it
                if caption.split()[-1] == 'end':
                    all_candidates.append([caption, score])
                    continue

                seq = tokenizer.texts_to_sequences([caption])[0]
                seq = pad_sequences([seq], maxlen=max_length, padding='post')

                yhat = model.predict([np.expand_dims(feature, axis=0), seq], verbose=0)
                

                # Select top `beam_index` most probable next words (highest logits)
                top_indices = np.argsort(yhat[0])[-beam_index:]

                for idx in top_indices:
                    word = tokenizer.index_word.get(idx)
                    if word is None:
                        continue

                    new_caption = caption + ' ' + word

                    # Add log probability to cumulative score
                    # 1e-10 prevents log(0)
                    new_score = score + np.log(yhat[0][idx] + 1e-10)

                    all_candidates.append([new_caption, new_score])

            # Keep top `beam_index` candidates with highest score
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_index]

            # Exit if all top candidates end with 'end'
            if all([cap.split()[-1] == 'end' for cap, _ in sequences]):
                break

        # The best sequence is the one with the highest score and they are already sorted so the first one
        best_caption = sequences[0][0]

        return best_caption.replace('start', '').replace('end', '').strip()

    results = {}
    for img_name, feature in features.items():
        caption = beam_search(model, tokenizer, feature, max_length, beam_index=3)
        results[img_name] = caption

    return results