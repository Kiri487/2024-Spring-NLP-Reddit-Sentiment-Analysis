import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def build_vocab(sentences):
    vocab = {'PAD': 0, 'UNK': 1}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode_sentences(sentences, vocab, max_len):
    encoded = []
    for sentence in sentences:
        encoded_sentence = [vocab.get(word, vocab['UNK']) for word in sentence]
        if len(encoded_sentence) < max_len:
            encoded_sentence += [vocab['PAD']] * (max_len - len(encoded_sentence))
        else:
            encoded_sentence = encoded_sentence[:max_len]
        encoded.append(encoded_sentence)
    return np.array(encoded)


def read_data(filename):
    data = pd.read_csv('./' + filename)
    lemmatizer = WordNetLemmatizer()
    sentences = []
    tags = []
    for index, row in data.iterrows():
        label = row['category']
        text = row['clean_comment']
        if not isinstance(text, str):
            text = ""
        text = re.sub(r'@\S+', '', text)  # Remove @user
        text = re.sub(r'http\S+', '', text)  # Remove link
        text = text.strip()
        words = word_tokenize(text)
        if all(word.strip(string.punctuation).isalpha() for word in words):
            lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]
            sentences.append(lemmatized_words)
            tags.append(label)
    return sentences, tags

def save_vocab_to_txt(vocab, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, index in sorted(vocab.items(), key = lambda item: item[1]):
            file.write(f"{word}\t{index}\n")

sentences, tags = read_data('Reddit_Data.csv')

print("Total sentences:", len(sentences))
print("Sample sentence:", sentences[0])
print("Corresponding tag:", tags[0])

index = np.arange(len(sentences))
np.random.shuffle(index)
train_size = int(0.8 * len(sentences))
train_index = index[:train_size]
test_index = index[train_size:]
train_sentences = [sentences[i] for i in train_index]
train_tags = [tags[i] for i in train_index]
test_sentences = [sentences[i] for i in test_index]
test_tags = [tags[i] for i in test_index]

print("Training sentences:", len(train_sentences))
print("Testing sentences:", len(test_sentences))

vocab = build_vocab(train_sentences)
save_vocab_to_txt(vocab, 'vocabulary.txt')
max_len = max(len(sentence) for sentence in train_sentences)

train_encoded = encode_sentences(train_sentences, vocab, max_len)
test_encoded = encode_sentences(test_sentences, vocab, max_len)

print("Encoded training data shape:", train_encoded.shape)
print("Encoded testing data shape:", test_encoded.shape)

np.savez('data.npz', train_data = train_encoded, train_tags = np.array(train_tags), test_data = test_encoded, test_tags = np.array(test_tags))