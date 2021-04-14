import os
import pickle
from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from ner_utils.pre_process import read_dataset, example_to_features, dict_from_input_data
import constants as c

import numpy as np


def convert_to_input(sentences, tags, tokenizer, label_map, max_seq_length):
    input_id_list, attention_mask_list, token_type_id_list = [], [], []
    label_id_list = []
    
    for x, y in tqdm(zip(sentences, tags), total=len(tags)):
        
        tokens = []
        label_ids = []
        seq_list = x.split(' ')
        seq_label_list = y.split(' ')
        
        for word, label in zip(seq_list, seq_label_list):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [label_map['[PAD]']] * (len(word_tokens) - 1))
        
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        
        label_ids = [label_map['[PAD]']] + label_ids + [label_map['[PAD]']]
        inputs = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=max_seq_length)
        
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_masks = [1] * len(input_ids)
        
        attention_mask_list.append(attention_masks)
        input_id_list.append(input_ids)
        token_type_id_list.append(token_type_ids)
        
        label_id_list.append(label_ids)
    
    return input_id_list, token_type_id_list, attention_mask_list, label_id_list


def retrieve_features(data_type, label_list, max_seq_length, tokenizer, label2id_pkl_file):
    label_map = {}
    X, y = read_dataset(os.path.join(c.PROCESSED_DATASET_DIR, data_type))
    
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(label2id_pkl_file, 'wb') as w:
        pickle.dump(label_map, w)
    input_id_list, token_type_id_list, attention_mask_list, label_id_list = convert_to_input(X, y, tokenizer, label_map,
                                                                                             max_seq_length)
    input_ids = pad_sequences(input_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    token_ids = pad_sequences(token_type_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    attention_masks = pad_sequences(attention_mask_list, maxlen=max_seq_length, dtype="long", truncating="post",
                                    padding="post")
    label_ids = pad_sequences(label_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    
    if data_type == c.TRAIN_FILE:
        return tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, token_ids,
                                                   label_ids)).map(example_to_features)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, token_ids,
                                                      label_ids)).map(example_to_features)
        inputs = []
        for idx, row in enumerate(input_ids):
            inputs.append(dict_from_input_data(row, attention_masks[idx], token_ids[idx]))
        
        # return dataset, inputs, np.reshape(label_ids, (len(label_ids)*c.MAX_SEQ_LENGTH, 1))
        return dataset, inputs, label_ids


def retrieve_pred_features(file_path, label_list, max_seq_length, tokenizer, label2id_pkl_file):
    label_map = {}
    X, y = read_dataset(file_path)
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(label2id_pkl_file, 'wb') as w:
        pickle.dump(label_map, w)
    input_id_list, token_type_id_list, attention_mask_list, label_id_list = convert_to_input(X, y, tokenizer, label_map,
                                                                                             max_seq_length)
    input_ids = pad_sequences(input_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    token_ids = pad_sequences(token_type_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    attention_masks = pad_sequences(attention_mask_list, maxlen=max_seq_length, dtype="long", truncating="post",
                                    padding="post")
    label_ids = pad_sequences(label_id_list, maxlen=max_seq_length, dtype="long", truncating="post",
                              padding="post")
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, token_ids,
                                                  label_ids)).map(example_to_features)
    inputs = []
    for idx, row in enumerate(input_ids):
        inputs.append(dict_from_input_data(row, attention_masks[idx], token_ids[idx]))
    
    # return dataset, inputs, np.reshape(label_ids, (len(label_ids)*c.MAX_SEQ_LENGTH, 1))
    return dataset, inputs, label_ids
