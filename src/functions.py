import string
from PIL import Image
import keras
import numpy as np
import os

# Create function to load the files
def load_file(filename:str):
    """
    Takes a filename and loads its contents into a string

    Args: 
        filename (string): path of the file that will be loaded

    Returns:
        text (string): a string of all of the lines in the filename
    """

    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    return text


# Create a function to seperate each image with its captions
def img_captions(filename:str):
    """
    Takes a file of images and all of its captions and creates a dictionary with each image 
                  as the key and the values are all of its  captions

    Args:
        filename (string): the name of the file with the images and captions pair

    Returns:
        image_descriptions (dictionary): a dictionary with each image as the key and its captions in a list as the value
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
    Takes a dictionary of image and its captions and removes punctuations, numbers, and changes uppercase letters into lowercase.

    Args: 
        captions (dictionary): A dictionary of each image as the key and a list of captions as the value.

    Returns:
        captions (dictionary): Returns the same dictionary after simplifying the captions 
    """
    
    table = str.maketrans("", "", string.punctuation)
    
    for img, img_captions in captions.items():
        for i, caption in enumerate(img_captions):
            
            caption.replace("-"," ") # Replacing "-" with a blank space
            words = caption.split() # Splitting each word in the dictionary

            words = [word.lower() for word in words] 
            words = [word.translate(table) for word in words] # Applying the translation table of removing the punctuation marks

            words = [word for word in words if len(word)>1] # Only keeping words that are more then 1 character long
            words = [word for word in words if word.isalpha()] # Removing numbers

            caption = " ".join(words)
            captions[img][i] = caption

    return captions


# Create a function to make up the vocab used in the text
def make_vocab(captions:dict):
    """
    Takes a dictionary of image and its captions and finds all unique words used to make up a set of vocabulary

    Args:
        captions (dictionary): A dictionary of each image as the key and a list of captions as the value.

    Returns:
        vocab (set): A set of unique words from all the captions
    """

    vocab = set()

    for key in captions.keys():
        
        [vocab.update(c.split()) for c in captions[key]]

    return vocab    


# Create a function that saves the images and its new editted captions
def save_captions(captions:dict, filename:str):
    """
    Takes the new dictionary of image and its new editted captions and saves them 
                   in the original formate of image - caption_number as a text file

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
    

def extract_features(directory):
    """
    Takes in a path for the images directory and returns a dictionary of each image and its feature vector that is extraced using the Xception model.

    Args:
        directory (string): A path of the directory where the raw image files are located.

    Returns:
        features (dictionary): A dictionary of each image and its feature vector.
    """
    
    xception_feature_extraction_model = keras.applications.Xception(include_top=False, pooling='average') # Don't want the final layers which output the classification prediction instead we stop at the raw features detected
    features = {}
    
    for img in os.listdir(directory):
        
        filename = directory / img
        
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0) # changing the shape of the image to (1, 299, 299, 3) as keras expect the input in batches even if batch of 1.
        image = image / 127.5
        image = image - 1.0
        
        feature = xception_feature_extraction_model.predict(image) # outputs the feature vector
        features[img] = feature
        
    return features