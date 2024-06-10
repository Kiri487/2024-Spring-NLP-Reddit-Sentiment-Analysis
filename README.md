# README 
## Environment
- Ubuntu 16.04.7 LTS
- NVIDIA GeForce RTX 3060（12 GB）
- CUDA 10.0.130
- Python 3.11.9
    - numpy 1.26.4
    - pandas 2.2.2
    - nltk 3.8.1
    - torch 2.3.1
    - scikit-learn 1.5.0
    - tqdm 4.66.4
    - transformers 4.41.2
    - wandb 0.17.1

## /code
- dataProcess.py
    - Handles data preprocessing tasks including tokenization, lemmatization, and encoding. 
    - It prepares the dataset for model training by converting text into a format suitable for neural network processing.
- getComment.py
    - Use the PRAW library to fetch comments from specified subreddits on Reddit.
- getResult.py
    - Analyzes the prediction results from the models, counting occurrences of sentiment tags and calculating averages for each file in the prediction directory.
- test_BERT.py
    - Executes sentiment prediction using the pre-trained BERT model on new data and saves the predictions to specified directories.

## /code/model
- model_BERT.py
    - Training code for the BERT model.
- model_LSTM.py
    - Training code for the Bi-LSTM model.