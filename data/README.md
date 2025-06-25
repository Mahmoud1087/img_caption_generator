#  data/ Directory

This directory contains all datasets and intermediate data used in this project. The data is organized into the following subdirectories:

- `raw/`: Original, unmodified data files as downloaded.
- `cleaned/`: Data that has been cleaned and preprocessed for modeling.
- `processed/`: Data that has undergone further transformations or feature extraction.

---

## 1. Data Source

The primary dataset used in this project is the **Flickr8k dataset**, introduced in:

> **M. Hodosh, P. Young, and J. Hockenmaier** (2013).  
> *"Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics."*  
> Journal of Artificial Intelligence Research, Volume 47, pages 853–899.  
> [Link to paper](http://www.jair.org/papers/paper3994.html)

This dataset includes a collection of 8,000 images along with multiple textual captions per image. Text files for training, validation, and testing splits are included.

---

## 2. Data Transformations

The following scripts under the `scripts/` directory are responsible for transforming raw data into cleaned or processed versions:

| Script            | Input File                                        | Output File                               | Description                                         |
|-------------------|---------------------------------------------------|--------------------------------------------|-----------------------------------------------------|
| `clean_data.py`   | `raw/Flickr8k_text/Flickr8k.token.txt`            | `cleaned/Flickr8k.token_cleaned.txt`       | Cleans image captions by lowercasing, removing punctuation, and normalizing text. |

Additional transformation scripts will be added and documented here as the project progresses.

---

## 3. Purpose of Files in `cleaned/`

| File                                | Purpose                                                                 | Related Notebook                     |
|-------------------------------------|-------------------------------------------------------------------------|--------------------------------------|
| `Flickr8k.token_cleaned.txt`        | A cleaned version of the original caption file. Prepares text for tokenization and model training. | Referenced in `01-image_recognition.ipynb` |

---

