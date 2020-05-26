import os
from bert import tokenization
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from bert import optimization
from bert import run_classifier
from keras.preprocessing.sequence import pad_sequences
import time

MAX_SEQ_LENGTH = 75
label_dict = {"B-NAME": 0, "B-LOC": 1,  "B-ORG": 2, "B-MISC": 3, "O": 4}


class RetrieveSentence(object):
    
    def __init__(self, data):
        self.data = data
        print(data[0])
        fn = lambda s: [(w, p, t) for w, p, t in zip(s[0], s[1], s[2])]
        self.grouped = self.data.apply(fn)
        self.sentences = [s for s in self.grouped]
    
    def retrieve(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def create_tokenizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_file = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12", "vocab.txt")
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


def load_and_split_data(tokenizer):
    input_data = pd.read_csv('labelled_ner.txt', sep="\t", header=None)
    # unique_TAGs = list(set(input_data[2]))
    # unique_TAGs_idx = {t: i for i, t in enumerate(unique_TAGs)}
    # unique_POS = list(set(input_data[1]))
    print("################################")
    sentences, labels = [], []
    sequence, sequence_labels = '', []
    for idx, word in enumerate(input_data[0]):
        sequence += ' ' + word
        sequence_labels.append(input_data[2][idx])
        if word == '.':
            sentences.append(sequence)
            labels.append(sequence_labels)
            sequence = ''
            sequence_labels = []
    # tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # X = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    #                   maxlen=MAX_SEQ_LENGTH, dtype="long", truncating="post", padding="post")
    # y = pad_sequences([[unique_TAGs_idx.get(l) for l in lab] for lab in unique_TAGs],
    #                   maxlen=MAX_SEQ_LENGTH, value=unique_TAGs_idx["O"], padding="post",
    #                   dtype="long", truncating="post")
    # y = pad_sequences([[label_dict.get(l) for l in lab] for lab in labels],
    #                   maxlen=MAX_SEQ_LENGTH, value=label_dict["O"], padding="post",
    #                   dtype="long", truncating="post")
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    # train_tokens = map(tokenizer.tokenize, X_train)
    # train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    # train_token_id = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    #
    # train_token_id = map(lambda tids: tids + [0] * (MAX_SEQ_LENGTH - len(tids)), train_token_id)
    # train_token_id = np.array(list(train_token_id))
    #
    # test_tokens = map(tokenizer.tokenize, X_test)
    # test_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], test_tokens)
    # test_token_id = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
    #
    # test_token_id = map(lambda tids: tids + [0] * (MAX_SEQ_LENGTH - len(tids)), test_token_id)
    # test_token_id = np.array(list(test_token_id))
    #
    # val_tokens = map(tokenizer.tokenize, X_val)
    # val_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], val_tokens)
    # val_token_id = list(map(tokenizer.convert_tokens_to_ids, val_tokens))
    #
    # val_token_id = map(lambda tids: tids + [0] * (MAX_SEQ_LENGTH - len(tids)), val_token_id)
    # val_token_id = np.array(list(val_token_id))
    return X_train, y_train, X_val, y_val, X_test, y_test, list(set(input_data[2]))


def set_output_dir():
    OUTPUT_DIR = './output'
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))


def preprocess_data():
    X_train, y_train, X_val, y_val, X_test, y_test, unique_labels = load_and_split_data(create_tokenizer())
    set_output_dir()
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train = [run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y_train[i]) for i, x in enumerate(X_train)]
    val = [run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y_val[i]) for i, x in enumerate(X_val)]
    test = [run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y_test[i]) for i, x in enumerate(X_test)]
    # train = np.apply_along_axis(lambda x, y: run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y), 0,
    #                             X_train, y_train)
    # val = np.apply_along_axis(lambda x, y: run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y), 0,
    #                           X_val, y_val)
    # test = np.apply_along_axis(lambda x, y: run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=None),
    #                            0, X_test, y_test)
    tokenizer = create_tokenizer()
    print(type(train), len(train))
    # print("TRAIN:", train, y_train)
    train_features = run_classifier.convert_examples_to_features(train, unique_labels, MAX_SEQ_LENGTH, tokenizer)
    val_features = run_classifier.convert_examples_to_features(val, unique_labels, MAX_SEQ_LENGTH, tokenizer)
    test_features = run_classifier.convert_examples_to_features(test, unique_labels, MAX_SEQ_LENGTH, tokenizer)
    return train_features, val_features, test_features


train_features, val_features, test_features = preprocess_data()
print("TRAIN_FEAT:", len(train_features))
