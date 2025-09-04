*Image Caption Generator*

Welcome to the Image Caption Generator repository!
This project uses deep learning with CNNs, LSTMs, and GloVe embeddings to automatically generate descriptive captions for images.

**Project Overview**

This repository implements a neural image captioning model that combines:

A Convolutional Neural Network (CNN) for feature extraction (image encoder).

A Recurrent Neural Network (LSTM) for caption generation (text decoder).

GloVe embeddings for better semantic understanding of words.

Beam search for improved caption inference.

The goal is to build a system that can look at an image and produce a natural language description.

**Data Sources**

- Flickr8k Dataset: Contains 8,000 images, each paired with five captions.

    This project uses data made by M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
    http://www.jair.org/papers/paper3994.html.

- Pre-trained GloVe Embeddings: (50d, 100d, 200d, 300d) used for embedding initialization.

    https://nlp.stanford.edu/projects/glove/

**How It Works**

- Data Cleaning: Text captions are preprocessed (lowercased, punctuation removed).

- Tokenization: Captions are tokenized, and sequences are padded.

- Feature Extraction: CNN extracts image feature vectors, saved in features.h5.

- Model Training: CNN + LSTM + GloVe embeddings are trained using categorical cross-entropy loss.

- Caption Generation: Uses greedy search and beam search strategies.

- Evaluation: Model performance measured using loss values and qualitative caption accuracy.


**Dependencies**

- tensorflow
- keras
- numpy
- matplotlib
- h5py
- PIL

**File Structure**

- data/                # Raw and cleaned Flickr8k dataset
- models/              # Saved models (baseline, model_1, model_2, …)
- notebooks/           # Jupyter notebooks (main development + GloVe files)
- scripts/             # Data cleaning, feature extraction, tokenization, training scripts
- src/                 # Config and utility functions


**Usage**

1. Preprocess Data
    
    ```
    python scripts/clean_data.py
    python scripts/tokenize.py
    ```

2. Extract Features

    `python scripts/feature_extraction.py`

3. Train Model

    `python scripts/main.py`

4. Generate Captions

    `from scripts.model import generate_caption`
    `caption = generate_caption("data/raw/testing_images/example.jpg")`
    `print("Generated Caption:", caption)`

5. Expected output:

    `Generated Caption:`
    `"a young boy is playing with a dog in the park"`


**Model Performance**

- Current training loss: ~2.5 (categorical cross-entropy).

- Qualitative results: Captions are semantically correct but need improvement in fluency.

- Beam search significantly improves caption quality compared to greedy decoding.

**Future Improvements**

- Add attention mechanism for better image–text alignment.

- Experiment with transformer-based models (e.g., ViT + GPT-style decoder).

- Fine-tune CNN backbone (currently pre-trained).

- Evaluate captions using BLEU / METEOR scores.
