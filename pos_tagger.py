from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
from itertools import tee
from pos_tagger_unknown import POSTagger_MLP
from tagger_constants import *
import csv
import logging
import string


# Configure logging
logging.basicConfig(level=logging.INFO)

""" Contains the part of speech tagger class. """


def evaluate(data, model, processes=8, cm_path="cm.png", beam_width=1):
    """
    Evaluate POS model with optional beam search.

    Args:
        data: Evaluation data.
        model: Trained POSTagger model.
        processes (int): Number of processes for parallel evaluation.
        cm_path (str): Path to save the confusion matrix.
        beam_width (int): Beam width for beam search. Defaults to 1 (greedy decoding).

    """
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n // processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum(
        [1 for s in sentences for w in s if w not in model.word2idx.keys()]
    )
    predictions = {i: None for i in range(n)}
    probabilities = {i: None for i in range(n)}

    start = time.time()
    pool = Pool(processes=processes)
    res = []

    if beam_width == None:
        logging.info("Evaluating with Viterbi")
    elif beam_width >= 2:
        logging.info(f"Evaluating with Beam-{beam_width}")
    else:
        logging.info(f"Evaluating with Greedy")

    for i in range(0, n, k):
        res.append(
            pool.apply_async(
                infer_sentences, [model, sentences[i : i + k], i, beam_width]
            )
        )
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(
            pool.apply_async(
                compute_prob, [model, sentences[i : i + k], tags[i : i + k], i]
            )
        )
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")

    token_acc = (
        sum(
            [
                1
                for i in range(n)
                for j in range(len(sentences[i]))
                if tags[i][j] == predictions[i][j]
            ]
        )
        / n_tokens
    )
    unk_token_acc = (
        sum(
            [
                1
                for i in range(n)
                for j in range(len(sentences[i]))
                if tags[i][j] == predictions[i][j]
                and sentences[i][j] not in model.word2idx.keys()
            ]
        )
        / unk_n_tokens
    )
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, ".")
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += (
                1
                if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx]
                else 0
            )
            num_whole_sent += 1
            start_idx = end_idx + 1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc / num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values()) / n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))

    confusion_matrix(
        pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, cm_path
    )

    return (
        whole_sent_acc / num_whole_sent,
        token_acc,
        unk_token_acc,
        sum(probabilities.values()) / n,
    )
    # return whole_sent_acc / num_whole_sent, token_acc, sum(probabilities.values()) / n

def get_user_input(prompt, valid_options):
    """
    Prompt the user to select an option and validate the input.
    :param prompt: The message to display to the user.
    :param valid_options: A list of valid options.
    :return: The selected option if valid, or an error message.
    """
    while True:
        user_input = input(prompt).strip().upper()
        if user_input in valid_options:
            return user_input
        else:
            print(f"Error: '{user_input}' is not a valid option. Please try again.")

# Utility function to flatten the data for easier handling
def flatten_data(data_words, data_tags):
    """
    Flattens the data by converting a list of sentences (with words and tags) into
    a flat list of words and a flat list of corresponding tags.
    :param data_words: List of lists, where each sublist is a sentence of words
    :param data_tags: List of lists, where each sublist is a sentence of POS tags
    :return: A flat list of words and a flat list of corresponding tags
    """
    flattened_words = [
        word for sentence in data_words for word in sentence
    ]  # Flatten the word sentences
    flattened_tags = [
        tag for tag_list in data_tags for tag in tag_list
    ]  # Flatten the tag sentences
    return flattened_words, flattened_tags


def has_numbers(inputString):
    """
    Check if a string contains any digits.
    :param inputString: The string to check
    :return: Boolean indicating whether the string has numbers
    """
    return any(char.isdigit() for char in inputString)


def contains_dash(inputString):
    """
    Check if a string contains a dash ('-').
    :param inputString: The string to check
    :return: Boolean indicating whether the string has a dash
    """
    return "-" in inputString


def is_punctuation(word):
    """
    Check if a word is a punctuation mark.
    :param word: The word to check
    :return: Boolean indicating if the word is a punctuation
    """
    return word in string.punctuation


class POSTagger:
    """
    POS Tagger class for handling training, evaluation, and inference. Supports different
    N-gram models and smoothing techniques.
    """

    def __init__(self, inference_method, smoothing_method=None, beam_width=1):
        """
        Initializes the tagger model with inference and smoothing methods.
        :param inference_method: The method for decoding the POS tags (greedy, viterbi, etc.)
        :param smoothing_method: The smoothing technique (e.g., Laplace)
        """
        self.smoothing_method = smoothing_method  # Store the smoothing method
        self.inference_method = inference_method  # Store the inference method
        self.unigrams = None  # Placeholder for unigram probabilities
        self.bigrams = None  # Placeholder for bigram probabilities
        self.trigrams = None  # Placeholder for trigram probabilities
        self.lexical = None  # Placeholder for emission probabilities
        self.ngram = None  # N-gram type (unigram, bigram, trigram)
        self.num_words = -1  # Total number of words in the dataset
        self.beam_width = 1 # default tagger method (greedy if none other is provided)

    def get_unigrams(self):
        """
        Computes unigram probabilities for tags based on their occurrences in the data.
        """
        unigrams = [
            sum(x.count(tag) for x in self.data_tags) / self.num_words
            for tag in self.all_tags
        ]
        self.unigrams = np.array(
            unigrams
        )  # Store unigrams as numpy array for efficiency

    def get_bigrams(self):
        """
        Computes bigram probabilities for tags based on their co-occurrences in the data.
        """
        if self.unigrams is None:
            self.get_unigrams()
        bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))

        def pairwise(iterable):
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        for document in self.data_tags:
            for curr, next_word in pairwise(document):
                if self.smoothing_method == LAPLACE:
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (
                        LAPLACE_FACTOR + self.unigrams[self.tag2idx[curr]] * self.num_words
                    )
                else:
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (
                        self.unigrams[self.tag2idx[curr]] * self.num_words
                    )
        
        # Normalize only if smoothing method is not None
        if self.smoothing_method == LAPLACE:
            for i in range(len(bigrams)):
                num_log = np.log(LAPLACE_FACTOR) - np.log(len(self.all_tags))
                denom_log = np.log(self.unigrams[i] * self.num_words + LAPLACE_FACTOR)
                bigrams[i] += np.exp(num_log - denom_log)
        self.bigrams = bigrams

    def get_trigrams(self):
        """
        Computes trigram probabilities for tags based on their occurrences in the data.
        """
        if self.bigrams is None:
            self.get_bigrams()
        trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        bigram_denominators = np.zeros((len(self.all_tags), len(self.all_tags)))

        def triplewise(iterable):
            a, b, c = tee(iterable, 3)
            next(b, None)
            next(c, None)
            next(c, None)
            return zip(a, b, c)

        for document in self.data_tags:
            for curr, next_word, nextnext_word in triplewise(document):
                if self.smoothing_method == LAPLACE:
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1
                    bigram_denominators[self.tag2idx[curr], self.tag2idx[next_word]] += 1
                else:
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1 / (
                        self.bigrams[self.tag2idx[curr], self.tag2idx[next_word]] * self.unigrams[self.tag2idx[curr]] * self.num_words
                    )

        if self.smoothing_method == LAPLACE:
            trigrams = np.exp(np.log(trigrams + LAPLACE_FACTOR / len(self.all_tags)) - np.log(bigram_denominators[:, :, None] + LAPLACE_FACTOR))
        
        self.trigrams = trigrams

    def get_emissions(self):
        """
        Computes emission probabilities (Prob(word | tag)) by counting tag-word pairs.
        """
        lexical = np.zeros((len(self.all_tags), len(self.all_words)))  # Initialize lexical matrix
        for document_words, document_tags in zip(self.data_words, self.data_tags):
            for word, tag in zip(document_words, document_tags):
                lexical[
                    self.tag2idx[tag], self.word2idx[word]
                ] += 1  # Count occurrences of word given a tag
        tag_counts = np.sum(
            lexical, axis=1, keepdims=True
        )  # Count the total number of words per tag for normalization
        # unigram_probs = self.unigrams / np.sum(
        #     self.unigrams
        # )  # Normalize unigram probabilities
        if self.smoothing_method == LAPLACE:  # Apply Laplace smoothing
            # Apply Laplace smoothing
            #lexical += 1e-10
            #tag_counts += len(self.all_words)
            temp = np.log(LAPLACE_FACTOR) + np.log(1) - np.log(len(self.all_words))
            lexical = np.log(lexical + np.exp(temp)) - np.log(tag_counts + LAPLACE_FACTOR)  # Convert counts to log-probabilities to avoid underflow
        else:
            # Add a small constant even without Laplace smoothign to avoid log(0)
            lexical += 1e-10
            tag_counts += 1e-10
            lexical = np.log(lexical) - np.log(tag_counts)
            # lexical = np.log(lexical) - np.log(tag_counts)
        # lexical = np.log(lexical + np.exp(temp)) - np.log(
        #     tag_counts + LAPLACE_FACTOR
        # )  # Convert counts to log-probabilities to avoid underflow
        #lexical = np.log(lexical) - np.log(tag_counts)
        self.lexical = np.exp(lexical)  # Store the emission probabilities


    def train(self, data, ngram=2):
        """
        Trains the model by computing transition and emission probabilities from the data.
        :param data: The training data (sentences and tags)
        :param ngram: The N-gram model to use (e.g., unigram, bigram, trigram)
        """
        self.data = data  # Store the training data
        self.data_words = data[0]  # Extract words from the data
        self.data_tags = data[1]  # Extract tags from the data

        self.all_tags = list(
            set([t for tag in data[1] for t in tag])
        )  # Create a set of unique tags
        self.tag2idx = {
            self.all_tags[i]: i for i in range(len(self.all_tags))
        }  # Map tags to indices
        self.idx2tag = {
            v: k for k, v in self.tag2idx.items()
        }  # Reverse map from index to tag

        self.all_words = list(
            set([word for sentence in self.data_words for word in sentence])
        )  # Create a set of unique words
        self.word2idx = {
            self.all_words[i]: i for i in range(len(self.all_words))
        }  # Map words to indices
        self.idx2word = {
            v: k for k, v in self.word2idx.items()
        }  # Reverse map from index to word
        self.num_words = sum(
            len(d) for d in self.data_words
        )  # Count total number of words in the dataset

        self.ngram = ngram  # Store the N-gram type (unigram, bigram, or trigram)

        # Compute the unigram, bigram, and trigram probabilities
        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()

        # Apply the approporiate smoothing method
        if self.smoothing_method == LINEAR_INTERPOLATION:
            self.trigrams = self.linear_interpolation(
                self.unigrams, self.bigrams, self.trigrams
            )
        elif self.smoothing_method == LAPLACE:
            # Laplace smoothing is already applied
            pass
        else:
            pass

        # Train the MLP classifier for unknown words
        self.clf = POSTagger_MLP(
            data[0], model_type="MLP"
        )  # Initialize the MLP-based tagger for unknown words
        train_x, train_y = flatten_data(
            data[0], data[1]
        )  # Flatten the sentences and tags for training
        self.clf.train(
            np.array(train_x), np.array(train_y)
        )  # Train the MLP classifier on flattened data

    def sequence_probability(self, sequence, tags):
        """
        Computes the log-probability of a tagged sequence (sentence).
        :param sequence: The input sentence (sequence of words)
        :param tags: The corresponding POS tags for the sentence
        :return: The probability of the sequence given the tags
        """
        if self.trigrams is None:  # Ensure trigrams are computed
            self.get_trigrams()
        if self.lexical is None:  # Ensure emission probabilities are computed
            self.get_emissions()
        log_probability = 0  # Initialize log-probability
        prev_prev_tag = None  # Track the previous two tags for trigram models
        prev_tag = None
        for tag, word in zip(tags, sequence):  # Iterate over each word-tag pair
            if word not in self.word2idx.keys():  # Handle unknown words
                return 0
            log_probability += np.log(
                self.lexical[self.tag2idx[tag], self.word2idx[word]]
            )  # Add emission probability
            if self.ngram == 1:
                log_probability += np.log(
                    self.unigrams[self.tag2idx[tag]]
                )  # Add unigram transition probability
            elif self.ngram == 2:
                if prev_tag is None:
                    pass
                else:
                    log_probability += np.log(
                        self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]]
                    )  # Add bigram transition probability
            else:
                if prev_prev_tag is None:
                    if prev_tag is None:
                        pass
                    else:
                        #log_probability += np.log(
                        #    max(
                        #        self.lexical[self.tag2idx[tag], self.word2idx[word]],
                        #        1e-10,
                        #    )
                        #)
                        log_probability += np.log(
                            self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]]
                        )  # Use bigram transition
                else:
                    log_probability += np.log(
                        self.trigrams[
                            self.tag2idx[prev_prev_tag],
                            self.tag2idx[prev_tag],
                            self.tag2idx[tag],
                        ]
                    )  # Add trigram transition
            prev_prev_tag = prev_tag  # Update previous tags
            prev_tag = tag
        return np.exp(
            log_probability
        )  # Return the final probability by exponentiating the log-probability

    def inference(self, sequence, beam_width):
        """
        Tags a sequence with part-of-speech tags using the selected inference method.
        :param sequence: The input sentence (sequence of words)
        :param beam_width: The beam width to use for beam search. If beam_width == 1, greedy decoding is used.
        :return: The predicted POS tags for the sentence
        """    
        if self.inference_method == VITERBI:
            beam_width = None
            return self.viterbi(sequence)
        elif self.inference_method == BEAM_K:
            return self.beam_search(sequence, beam_width)
        else:
            beam_width = 1
            return self.greedy(sequence)


    def get_greedy_best_tag(self, word, prev_tag, prev_prev_tag):
        """
        Greedily selects the best POS tag for a word based on the current and previous tags.
        :param word: The word to tag
        :param prev_tag: The previous tag
        :param prev_prev_tag: The tag before the previous tag (for trigram models)
        :return: The best tag for the word, and the updated previous tags
        """
        best_tag = None
        if self.ngram == 1:  # For unigram models, select the most likely tag
            best_tag = self.idx2tag[
                np.argmax(self.lexical[:, self.word2idx[word]] * self.unigrams)
            ]
        elif self.ngram == 2:  # For bigram models, consider the previous tag
            if prev_tag is None:
                best_tag = "O"  # Default tag if no previous tag
            else:
                best_tag_index = np.argmax(
                    self.lexical[:, self.word2idx[word]]
                    * self.bigrams[self.tag2idx[prev_tag], :]
                )
                best_tag = self.idx2tag[best_tag_index]
            prev_tag = best_tag
        elif self.ngram == 3:  # For trigram models, consider the two previous tags
            if prev_tag is None:
                best_tag = "O"
            elif prev_prev_tag is None:
                best_tag_index = np.argmax(
                    self.lexical[:, self.word2idx[word]]
                    * self.bigrams[self.tag2idx[prev_tag], :]
                )
                best_tag = self.idx2tag[best_tag_index]
            else:
                best_tag_index = np.argmax(
                    self.lexical[:, self.word2idx[word]]
                    * self.trigrams[
                        self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :
                    ]
                )
                best_tag = self.idx2tag[best_tag_index]
            prev_prev_tag = prev_tag
            prev_tag = best_tag
        return best_tag, prev_tag, prev_prev_tag

    def greedy(self, sequence):
        """
        Greedy decoding for POS tagging. Decodes the most likely sequence of tags
        by greedily selecting the best tag for each word.
        :param sequence: The input sentence (sequence of words)
        :return: The predicted POS tags for the sentence
        """
        if self.lexical is None:
            self.get_emissions()  # Ensure emission probabilities are computed
        if self.trigrams is None:
            self.get_trigrams()  # Ensure trigram probabilities are computed
        prev_prev_tag = None  # Track previous two tags for trigram models
        prev_tag = None

        logging.debug(f"Decoding sequence of length {len(sequence)}")
        result = []
        for i, word in enumerate(sequence):
            best_tag = None
            if word not in self.word2idx:
                best_tag = self.clf.predict_word(word)  # Predict tag for unknown words
            else:
                best_tag, prev_tag, prev_prev_tag = self.get_greedy_best_tag(
                    word, prev_tag, prev_prev_tag
                )  # Use greedy approach to get best tag
            result.append(best_tag)
        return result  # Return the predicted sequence of tags

    def linear_interpolation(self, unigrams, bigrams, trigrams):
        """
        Apply linear interpolation to smooth the trigram model.
        Combines unigrams, bigrams, and trigrams with weights (lambdas).
        :param unigrams: Unigram probabilities.
        :param bigrams: Bigram probabilities.
        :param trigrams: Trigram probabilities.
        :param lambdas: Tuple of lambda weights (lambda1, lambda2, lambda3) for smoothing.
        :return: Smoothed trigram probabilities.
        """
        lambda1, lambda2, lambda3 = LAMBDAS_LINEAR_INTERPOLATION
        smoothed_trigrams = np.zeros_like(trigrams)

        for i in range(trigrams.shape[0]):
            for j in range(trigrams.shape[1]):
                for k in range(trigrams.shape[2]):
                    trigram_prob = trigrams[i, j, k]
                    bigram_prob = bigrams[j, k]
                    unigram_prob = unigrams[k]

                    smoothed_prob = (
                        lambda1 * unigram_prob
                        + lambda2 * bigram_prob
                        + lambda3 * trigram_prob
                    )
                    smoothed_trigrams[i, j, k] = smoothed_prob

        return smoothed_trigrams

    def beam_search(self, sequence, beam_width=3):
        """
        Beam search decoding for POS tagging. Instead of selecting the single best tag at each step (like greedy),
        beam search keeps track of the 'beam_width' best tag sequences.

        :param sequence: The input sentence (sequence of words)
        :param beam_width: The number of top paths to explore at each step
        :return: The predicted POS tags for the sentence
        """
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
            
        if self.lexical is None or self.trigrams is None:
            logging.error("Model not trained or incomplete!")
            return []

        beams = [([], 0)]  # Initialize beams with log probability of 0

        for word in sequence:
            new_beams = []
            if word not in self.word2idx:
                best_tag = self.clf.predict_word(word)
                for seq, log_prob in beams:
                    new_seq = seq + [best_tag]
                    new_log_prob = log_prob  # No change in probability
                    new_beams.append((new_seq, new_log_prob))
            else:
                for seq, log_prob in beams:
                    prev_tag = seq[-1] if seq else None
                    prev_prev_tag = seq[-2] if len(seq) > 1 else None
                    for tag in self.all_tags:
                        transition_log_prob = np.log(
                            self.compute_transition_prob(prev_prev_tag, prev_tag, tag)
                        )
                        emission_log_prob = np.log(
                            self.lexical[self.tag2idx[tag], self.word2idx[word]]
                        )
                        new_log_prob = (
                            log_prob + transition_log_prob + emission_log_prob
                        )
                        new_beams.append((seq + [tag], new_log_prob))

            # Sort beams by log probability and keep the top 'beam_width'
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams

        return beams[0][0]

    def compute_transition_prob(self, prev_prev_tag, prev_tag, tag):
        """
        Compute transition probability for n-gram model with smoothing.
        """
        if self.ngram == 1:
            # Unigram: return the unigram probability with a small constant for smoothing
            return max(self.unigrams[self.tag2idx[tag]], 1e-10)
        elif self.ngram == 2:
            if prev_tag is None:
                return 1.0
            # Return bigram probability with a small constant
            return max(self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]], 1e-10)
        elif self.ngram == 3:
            if prev_prev_tag is None or prev_tag is None:
                if prev_tag is None:
                    return 1.0
                # Fallback to bigram with smoothing
                return max(
                    self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]], 1e-10
                )
            # Trigram probability with a small constant for smoothing
            return max(
                self.trigrams[
                    self.tag2idx[prev_prev_tag],
                    self.tag2idx[prev_tag],
                    self.tag2idx[tag],
                ],
                1e-10,
            )

    def compare_greedy_beam(dev_data, model):
        # Run both greedy and beam search
        greedy_results = infer_sentences(model, dev_data[0], 0, beam_width=1)
        beam_results = infer_sentences(
            model, dev_data[0], 0, beam_width=3
        )  # k=3 for beam search

        # Log differences and calculate total improvement
        differences = 0
        improvements = 0
        for i in range(len(greedy_results)):
            if greedy_results[i] != beam_results[i]:
                differences += 1
                if (
                    beam_results[i] == dev_data[1][i]
                ):  # Check if beam result matches gold standard
                    improvements += 1
                logging.info(
                    f"Sentence {i}: greedy={greedy_results[i]}, beam={beam_results[i]}"
                )

        logging.info(f"Total differences: {differences}")
        logging.info(f"Total improvements from beam search: {improvements}")
        
    def viterbi_bigram(self, sequence):
        """
        Viterbi algorithm for bigram POS tagging. Decodes the most likely sequence of tags
        using dynamic programming and bigram transitions.
        :param sequence: The input sequence (list of words)
        :return: The predicted POS tags for the sequence
        """
        decoded_tags = [None] * len(sequence)
        scores = np.full((len(self.all_tags), len(sequence)), -np.inf)
        backpointers = np.full((len(self.all_tags), len(sequence)), None)

        scores[self.tag2idx['O'], 0] = 0  # Initialization for start tag
        
        # Loop through each position in the sequence (i.e., each word in the sentenc
        for t in range(1, len(sequence)):
            if sequence[t] not in self.word2idx:
                # If the word is unknown, predict its tag using the unknown word classifier
                predicted_tag_idx = self.tag2idx[self.clf.predict_word(sequence[t])]
                # Update scores and backpointers based on the predicted tag
                scores[predicted_tag_idx, t] = np.max(scores[:, t - 1] + np.log(self.bigrams[:, predicted_tag_idx]))
                backpointers[predicted_tag_idx, t] = np.argmax(scores[:, t - 1] + np.log(self.bigrams[:, predicted_tag_idx]))
                continue
            # If the word is known, loop through all possible tags
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag] # Get index for the current tag
                emission_score = self.lexical[tag_idx, self.word2idx[sequence[t]]] # Get emission probability for word|tag
                # Find the best previous tag and its score, updating scores and backpointers
                best_prev_tag_idx = np.argmax(scores[:, t - 1] + np.log(self.bigrams[:, tag_idx]))
                best_prev_score = np.max(scores[:, t - 1] + np.log(self.bigrams[:, tag_idx]))
                scores[tag_idx, t] = np.log(emission_score) + best_prev_score
                backpointers[tag_idx, t] = best_prev_tag_idx

        # Start with the final word and find the highest scoring tag
        final_tag_idx = np.argmax(scores[:, len(sequence) - 1])
        decoded_tags[len(sequence) - 1] = self.idx2tag[final_tag_idx]
        prev_tag_idx = int(backpointers[final_tag_idx, len(sequence) - 1])

        # Trace back from the final word to the first, following the backpointers to get the optimal tag path
        for t in range(len(sequence) - 2, 0, -1):
            decoded_tags[t] = self.idx2tag[prev_tag_idx]
            prev_tag_idx = int(backpointers[prev_tag_idx, t])

        # Set the tag for the first word
        decoded_tags[0] = self.idx2tag[prev_tag_idx]
        return decoded_tags


    def viterbi_trigram(self, sequence):
        """
        Viterbi algorithm for trigram POS tagging. Decodes the most likely sequence of tags
        using dynamic programming and trigram transitions.
        :param sequence: The input sequence (list of words)
        :return: The predicted POS tags for the sequence
        """
        NUM_TAGS = len(self.all_tags)
        decoded_tags = [None] * len(sequence)

        scores = np.full((NUM_TAGS * NUM_TAGS, len(sequence)), -np.inf)
        backpointers = np.full((NUM_TAGS * NUM_TAGS, len(sequence)), -np.inf)

        start_idx = self.tag2idx['O']
        scores[start_idx * NUM_TAGS: (start_idx + 1) * NUM_TAGS, 0] = 0  # Initialization

        for t in range(1, len(sequence)):
            fixed_tag = None
            if sequence[t] not in self.word2idx:
                fixed_tag = self.clf.predict_word(sequence[t])

            for tag in self.all_tags:
                curr_tag = tag if fixed_tag is None else fixed_tag
                curr_tag_idx = self.tag2idx[curr_tag]
                emission_score = 1 if fixed_tag is not None else self.lexical[curr_tag_idx, self.word2idx[sequence[t]]]
                log_emission_score = np.log(emission_score)

                for prev_tag in self.all_tags:
                    prev_tag_idx = self.tag2idx[prev_tag]
                    index = prev_tag_idx * NUM_TAGS + curr_tag_idx
                    best_prev_score = np.max(scores[prev_tag_idx::NUM_TAGS, t - 1] + np.log(self.trigrams[:, prev_tag_idx, curr_tag_idx]))
                    best_prev_tag = np.argmax(scores[prev_tag_idx::NUM_TAGS, t - 1] + np.log(self.trigrams[:, prev_tag_idx, curr_tag_idx]))
                    scores[index, t] = log_emission_score + best_prev_score
                    backpointers[index, t] = best_prev_tag

        last_tag, second_last_tag = None, None
        max_final_score = -np.inf
        
        # Find the best ending pair of tags (second-to-last and last tags)
        for i in range(NUM_TAGS):
            final_score = np.max(scores[i::NUM_TAGS, len(sequence) - 1] + self.trigrams[:, i, self.tag2idx['.']])
            entry_idx = np.argmax(scores[i::NUM_TAGS, len(sequence) - 1] + self.trigrams[:, i, self.tag2idx['.']])

            if final_score > max_final_score:
                max_final_score = final_score
                last_tag = self.idx2tag[i]
                second_last_tag = self.idx2tag[entry_idx]

        # Assign the final two tags to the last two words of the sequence
        decoded_tags[len(sequence) - 1] = last_tag
        decoded_tags[len(sequence) - 2] = second_last_tag

        # Trace back the rest of the sequence using the backpointers
        for t in range(len(sequence) - 3, 0, -1):
            if t == 0:
                break
            prev_tag_idx = int(backpointers[self.tag2idx[decoded_tags[t + 2]] + NUM_TAGS * self.tag2idx[decoded_tags[t + 1]], t + 2])
            decoded_tags[t] = self.idx2tag[prev_tag_idx]

        decoded_tags[0] = self.idx2tag[start_idx]
        return decoded_tags


    def viterbi(self, sequence):
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
        if self.lexical is None or self.trigrams is None:
            logging.error("Model not trained or incomplete!")
            return []
        if self.ngram == 2: 
            return self.viterbi_bigram(sequence)
        elif self.ngram == 3:
            return self.viterbi_trigram(sequence)


if __name__ == "__main__":
    inference_method = get_user_input(
        "Please select the inference method (GREEDY, BEAM_K, VITERBI): ",
        ["GREEDY", "BEAM_K", "VITERBI"],
    )

    # Set beam_width based on inference method
    beam_width = None
    if inference_method == "GREEDY":
        beam_width = 1
    elif inference_method == "VITERBI":
        beam_width = None
    elif inference_method == "BEAM_K":
        # If BEAM_K is selected, prompt the user to enter an integer for k
        while True:
            try:
                beam_width = int(input("Please enter an integer value for k (beam width): "))
                break
            except ValueError:
                print("Error: Please enter a valid integer for k.")

    # Prompt user to select the smoothing method
    smoothing_method = get_user_input(
        "Please select the smoothing method (LAPLACE, LINEAR_INTERPOLATION): ",
        ["LAPLACE", "LINEAR_INTERPOLATION"],
    )

    # Prompt user to select N-gram type (bigram or trigram)
    while True:
        try:
            ngram = int(input("Please enter 2 for bigram or 3 for trigram: "))
            if ngram in [2, 3]:
                break
            else:
                print("Error: Please select 2 for bigram or 3 for trigram.")
        except ValueError:
            print("Error: Please enter a valid integer (2 or 3).")

    # Instantiate the POS tagger with the selected options
    pos_tagger = POSTagger(
        inference_method=inference_method,
        smoothing_method=smoothing_method,
        beam_width=beam_width,
    )

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    
    # Train the POS tagger
    pos_tagger.train(train_data, ngram=ngram) #change n-gram to 2 or 3 for bigram or trigram respectively

    evaluate(dev_data, pos_tagger, beam_width = beam_width)

    # # List to store the results
    # results = []

    # # Define the different configurations for evaluation
    # configs = [
    #     {"smoothing": LAPLACE, "beam_width": 1, "description": "Greedy with Laplace"},
    #     {
    #         "smoothing": LINEAR_INTERPOLATION,
    #         "beam_width": 1,
    #         "description": "Greedy with Linear Interpolation",
    #     },
    #     {
    #         "smoothing": LINEAR_INTERPOLATION,
    #         "beam_width": 2,
    #         "description": "Greedy with Linear Interpolation",
    #     },
    #     {
    #         "smoothing": LINEAR_INTERPOLATION,
    #         "beam_width": 3,
    #         "description": "Beam search (k=3) with Linear Interpolation",
    #     },
    # ]

    # for config in configs:
    #     # Logging the current configuration
    #     logging.info(f"Running {config['description']}")

    #     # Initialize the POS tagger with the current configuration
    #     pos_tagger = POSTagger(
    #         inference_method=GREEDY,  # GREEDY method is fine for both greedy and beam search (beam_width changes behavior)
    #         smoothing_method=config["smoothing"],
    #     )

    #     # Train the POS tagger with 3-gram model
    #     pos_tagger.train(train_data, ngram=3)

    #     # Evaluate on the development set
    #     whole_sent_acc, token_acc, unk_token_acc, mean_prob = evaluate(
    #         dev_data, pos_tagger, beam_width=config["beam_width"]
    #     )

    #     # Store the results for this configuration
    #     results.append(
    #         {
    #             "description": config["description"],
    #             "whole_sentence_accuracy": whole_sent_acc,
    #             "token_accuracy": token_acc,
    #             "unk_token_accuracy": unk_token_acc,
    #             "mean_probability": mean_prob,
    #         }
    #     )

    # # After running all configurations, log or print the results
    # print("\nFinal Results:")
    # for result in results:
    #     print(f"{result['description']}:")
    #     print(f"  Whole sentence accuracy: {result['whole_sentence_accuracy']:.4f}")
    #     print(f"  Token accuracy: {result['token_accuracy']:.4f}")
    #     print(f"  Unknown token accuracy: {result['unk_token_accuracy']:.4f}")
    #     print(f"  Mean probability: {result['mean_probability']:.4f}")
    #     print("-" * 40)

    # # Train the POS tagger
    # pos_tagger.train(train_data, ngram=3)

    # evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence, beam_width = beam_width))
        # test_predictions.extend(pos_tagger.inference(sentence, beam_width=1))

    # Write them to a file to update the leaderboard
    with open("test_y.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "predicted_tag"])  # Write header
        for i, tag in enumerate(test_predictions):
            writer.writerow([i, tag])  # Write ID and predicted tag
