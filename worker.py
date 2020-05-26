from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_SEQ_LENGTH = 75
label_dict = {"B-NAME": 0, "B-LOC": 1, "B-ORG": 2, "B-MISC": 3, "O": 4}


def load_and_split_data():
    input_data = pd.read_csv('labelled_ner.txt', sep="\t", header=None)
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
    unique_labels = list(set(input_data[2]))
    return unique_labels, sentences
    # X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    # return X_train, y_train, X_val, y_val, X_test, y_test, list(set(input_data[2]))


def set_output_dir():
    OUTPUT_DIR = './output'
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))


max_seq_length = 75
current_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12")
vocab_file = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12", "vocab.txt")
config_json_file = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12", "bert_config.json")
unique_labels, sentences = load_and_split_data()
label_map = {}
for i, item in enumerate(unique_labels):
    label_map[i] = item
