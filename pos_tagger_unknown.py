import string
import time
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tagger_constants import *
from tagger_utils import *
import joblib
import logging

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags."""
    processes = 8  # Number of processes for parallel evaluation
    sentences = data[0]  # Extract sentences (words)
    words, tags = flatten_data(data[0], data[1])  # Flatten the data into words and tags
    n = len(words)  # Total number of words
    k = n // processes  # Batch size per process
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])  # Count unknown tokens
    words = [word for word in words if word not in model.word2idx.keys()]  # Filter unknown words
    tags = [tag for word, tag in zip(words, tags) if word not in model.word2idx.keys()]  # Filter corresponding tags
    predictions = {i: None for i in range(unk_n_tokens)}  # Initialize predictions

    start = time.time()
    pool = Pool(processes=processes)  # Parallel processing using multiple CPU cores
    res = []
    for i in range(0, n, k):
        # Apply model inference to each batch
        res.append(pool.apply_async(infer_sentences, [model, words[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]  # Collect results from workers
    predictions = dict()
    for a in ans:
        predictions.update(a)  # Update predictions with results
    print(f"Inference Runtime: {(time.time() - start) / 60} minutes.")  # Log inference time
    
    # Calculate accuracy for unknown tokens
    unk_token_acc = sum([1 for i in range(len(words)) if tags[i] == predictions[i] and words[i] not in model.word2idx.keys()]) / unk_n_tokens
    
    misclassification_count = 0
    for i in range(len(words)):
        # Output misclassified unknown words (up to 20)
        if words[i] not in model.word2idx.keys() and tags[i] != predictions[i]:
            print(f"Sentence: {i}, Word: {words[i]}, Actual Tag: {tags[i]}, Predicted Tag: {predictions[i]}")
            misclassification_count += 1
        if misclassification_count >= 20:
            break
    print("Unk token acc: {}".format(unk_token_acc))  # Print unknown token accuracy
    return

def flatten_data(data_words, data_tags):
    # Flatten nested lists of sentences and tags into single lists for easy processing
    flattened_words = [word for sentence in data_words for word in sentence]
    flattened_tags = [tag for tag_list in data_tags for tag in tag_list]
    return flattened_words, flattened_tags

# Utility functions to check if words contain specific properties
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def contains_dash(inputString):
    return '-' in inputString

def is_punctuation(word):
    return word in string.punctuation

class POSTagger_MLP():
    def __init__(self, documents, model_type="MLP"):
        """Initializes the tagger model parameters and anything else necessary."""
        # Initialize a TF-IDF vectorizer to transform words into numerical vectors
        self.vectorizer = TfidfVectorizer(max_features=300)  # Use max 300 features
        self.vectorizer.fit([' '.join(sentence) for sentence in documents])  # Fit on all sentences

        # Fit PCA to reduce the dimensionality of the embeddings
        all_embeddings = [self.vectorizer.transform([' '.join(sentence)]).toarray()[0] for sentence in documents]
        self.embeddings_new_length = 100  # Reduce dimensions to 100
        self.pca = PCA(n_components=self.embeddings_new_length).fit(all_embeddings)

        # Build the vocabulary and frequency table
        self.build_vocab(documents)
        freq = [sum(doc.count(word) for doc in documents) for word in self.vocab]  # Frequency of each word
        self.freq = np.array(freq)

        # Build suffix and prefix indices to capture word morphology
        self.build_suffix_indices(documents)
        self.build_prefix_indices(documents)
        self.model_type = model_type  # Either "MLP" or "Gradient Boosting"

    def build_vocab(self, documents):
        """Build a vocabulary and map words to indices. Filter words that occur more than 10 times."""
        # Create vocab by taking the unique words from all documents
        self.vocab = list(set([word for sentence in documents for word in sentence]))
        self.word2idx = {self.vocab[i]: i for i in range(len(self.vocab))}  # Map words to indices
        self.idx2word = {v: k for k, v in self.word2idx.items()}  # Reverse mapping

    def build_suffix_indices(self, documents):
        """Extract suffixes from words and map them to indices."""
        # Extract 3-gram, 2-gram, and 1-gram suffixes for each word and map them to indices
        self.tri_suffixes = list(set([word[-3:] for sentence in documents for word in sentence]))
        self.tri_suffixes.append('...')  # Add placeholder for unknown suffixes
        self.tri_suffix2idx = {self.tri_suffixes[i]: i for i in range(len(self.tri_suffixes))}
        self.idx2tri_suffix = {v: k for k, v in self.tri_suffix2idx.items()}

        # Similarly extract bi-suffixes and uni-suffixes
        self.bi_suffixes = list(set([word[-2:] for sentence in documents for word in sentence]))
        self.bi_suffixes.append('...')
        self.bi_suffix2idx = {self.bi_suffixes[i]: i for i in range(len(self.bi_suffixes))}
        self.idx2bi_suffix = {v: k for k, v in self.bi_suffix2idx.items()}

        self.uni_suffixes = list(set([word[-1:] for sentence in documents for word in sentence]))
        self.uni_suffixes.append('...')
        self.uni_suffix2idx = {self.uni_suffixes[i]: i for i in range(len(self.uni_suffixes))}
        self.idx2uni_suffix = {v: k for k, v in self.uni_suffix2idx.items()}

    # Functions to retrieve suffix indices for a given word (handling unknowns)
    def get_tri_suffix_index(self, word):
        try:
            return self.tri_suffix2idx[word[-3:]]
        except KeyError:
            return self.tri_suffix2idx['...']

    def get_bi_suffix_index(self, word):
        try:
            return self.bi_suffix2idx[word[-2:]]
        except KeyError:
            return self.bi_suffix2idx['...']
    
    def get_uni_suffix_index(self, word):
        try:
            return self.uni_suffix2idx[word[-1:]]
        except KeyError:
            return self.uni_suffix2idx['...']
        
    def build_prefix_indices(self, documents):
        """Extract prefixes from words and map them to indices."""
        # Same as suffixes but for prefixes (3, 2, 1-gram)
        self.tri_prefixes = list(set([word[:3] for sentence in documents for word in sentence]))
        self.tri_prefixes.append('...')
        self.tri_prefix2idx = {self.tri_prefixes[i]: i for i in range(len(self.tri_prefixes))}
        self.idx2tri_prefix = {v: k for k, v in self.tri_prefix2idx.items()}

        self.bi_prefixes = list(set([word[:2] for sentence in documents for word in sentence]))
        self.bi_prefixes.append('...')
        self.bi_prefix2idx = {self.bi_prefixes[i]: i for i in range(len(self.bi_prefixes))}
        self.idx2bi_prefix = {v: k for k, v in self.bi_prefix2idx.items()}

    # Retrieve prefix indices for a given word
    def get_tri_prefix_index(self, word):
        try:
            return self.tri_prefix2idx[word[:3]]
        except KeyError:
            return self.tri_prefix2idx['...']

    def get_bi_prefix_index(self, word):
        try:
            return self.bi_prefix2idx[word[:2]]
        except KeyError:
            return self.bi_prefix2idx['...']

    def predict_word(self, word):
        # Predict POS tag for an unknown word using MLP or Gradient Boosting model
        vector_word = self.get_word_vector(word)  # Get word's vector representation
        if self.model_type == "MLP":
            return self.clf.predict([vector_word])[0]
        else:
            prediction = self.clf.predict([vector_word])[0]
            return prediction
    
    def inference(self, word):
        # Wrapper around predict_word for inference
        vector_word = self.get_word_vector(word)
        if self.model_type == "MLP":
            return self.clf.predict([vector_word])[0]
        else:
            prediction = self.clf.predict([vector_word])[0]
            return prediction
    
    def get_word_vector(self, word):
        # Retrieve vector representation of the word using TF-IDF and PCA
        try:
            word_embedding = self.vectorizer.transform([word]).toarray()[0]  # Get TF-IDF vector
            reduced_embedding = self.pca.transform([word_embedding])[0]  # Reduce with PCA
            features = self.get_additional_features(word)  # Combine with additional features (suffix/prefix)
            return np.concatenate((features, reduced_embedding))  # Return combined feature vector
        except KeyError:
            return np.zeros(self.embeddings_new_length)  # Return zero vector if unknown

    def get_additional_features(self, word):
        # Extract morphological features of the word (suffixes, capitalization, etc.)
        features = {
            'tri_suffix_index': self.get_tri_suffix_index(word),
            'bi_suffix_index': self.get_bi_suffix_index(word),
            'uni_suffix_index': self.get_uni_suffix_index(word),
            'is_capitalized': int(word[0].isupper()),
            'num_capitals': sum([1 for char in word if char.isupper()]),
            'contains_dash': int('-' in word),
            'word_length': len(word),
            'contains_slash': int('\/' in word),
            'contains_number': int(any(char.isdigit() for char in word)),
        }
        return np.array(list(features.values()))  # Return as a numpy array of feature values

    def train(self, train_x, train_y):
        # Train the classifier (MLP or Gradient Boosting) for unknown words
        logging.info("Training unknown word model")
        filtered_words = []
        filtered_labels = []
        for word, label in zip(train_x, train_y):
            # Only train on words that occur less than 10 times (unknown word criteria)
            if self.freq[self.word2idx[word]] >= 10:
                pass
            else:
                vector = self.get_word_vector(word)  # Get word vector
                filtered_words.append(vector)
                filtered_labels.append(label)
        
        # Train MLP or Gradient Boosting classifier
        if self.model_type == "MLP":
            self.clf = MLPClassifier(hidden_layer_sizes=(100, 100, 50), max_iter=300)
            self.clf.fit(np.array(filtered_words), np.array(filtered_labels))
        else:
            self.clf = GradientBoostingClassifier()
            self.clf.fit(np.array(filtered_words), np.array(filtered_labels))
        logging.info("Unknown word model complete")

    def load_model(self, model_path):
        # Load a pre-trained model
        self.clf = joblib.load(model_path)

if __name__ == "__main__":
    # Load the training and development data
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    train_x, train_y = flatten_data(train_data[0], train_data[1])  # Flatten train data
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")  # Load dev data
    test_data = load_data("data/test_x.csv")  # Load test data
    
    # Initialize the unknown word tagger with MLP model
    clf = POSTagger_MLP(train_data[0], model_type="MLP")
    clf.train(np.array(train_x[:2000]), np.array(train_y[:2000]))  # Train the MLP model on a subset of data

    # Evaluate the model on the dev set
    evaluate(dev_data[:1000], clf)
