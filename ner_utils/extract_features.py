import collections
import logging
import pickle

import bert
import tensorflow as tf


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, mask, segment_ids, label_ids, is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def convert_seq_to_feature(data_row_idx, data_row, label_map, max_seq_length, tokenizer):
    text_list = data_row.text.split(' ')
    label_list = data_row.label.split(' ')
    tokens, labels = [], []
    for i, (word, label) in enumerate(zip(text_list, label_list)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        labels.append(label)
        for i, _ in enumerate(token):
            labels.append(label) if i == 0 else labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    padded_tokens = []
    input_type_ids = []
    label_ids = []
    padded_tokens.append("[CLS]")
    input_type_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        padded_tokens.append(token)
        input_type_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_word_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    input_mask = [1] * len(input_word_ids)
    # use zero to padding and you should
    while len(input_word_ids) < max_seq_length:
        input_word_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)
        label_ids.append(0)
        padded_tokens.append("[PAD]")
    assert len(input_word_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(input_type_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(padded_tokens) == max_seq_length
    if data_row_idx < 3:
        logging.info("*** Example ***")
        logging.info(f'guid: {data_row.guid}')
        logging.info(f'tokens: {" ".join([bert.bert_tokenization.printable_text(x) for x in tokens])}')
        logging.info(f'input_ids: {" ".join([str(x) for x in input_word_ids])}')
        logging.info(f'input_mask: {" ".join([str(x) for x in input_mask])}')
        logging.info(f'input_type_ids: {" ".join([str(x) for x in input_type_ids])}')
        logging.info(f'label_ids: {" ".join([str(x) for x in label_ids])}')
    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': tf.ones_like(input_word_ids).to_tensor(),
        'input_type_ids': tf.zeros_like(input_word_ids).to_tensor()
    }
    # we need no_tokens because if we do predict it can help us return to original token.
    return padded_tokens, label_ids, inputs


def convert_data_into_features(train_data, label_list, max_seq_length, tokenizer, tf_record_file, label2id_pkl_file):
    label_map = {}
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(label2id_pkl_file, 'wb') as w:
        pickle.dump(label_map, w)
    record_writer = tf.io.TFRecordWriter(tf_record_file)
    batch_tokens = []
    batch_labels = []
    batch_inputs = []
    for (row_idx, row_data) in enumerate(train_data):
        if row_idx % 5000 == 0:
            logging.info("Writing row_data %d of %d" % (row_idx, len(train_data)))
        padded_tokens, label_ids, inputs = convert_seq_to_feature(row_idx, row_data, label_map, max_seq_length,
                                                                  tokenizer)
        batch_tokens.extend(padded_tokens)
        batch_labels.extend(label_ids)
        batch_inputs.extend(inputs)
        
        record_writer.write(
            tf.train.Example(features=tf.train.Features(feature={
                'input_word_ids': create_int_feature(inputs['input_word_ids']),
                'input_mask': create_int_feature(inputs['input_mask']),
                'input_type_ids': create_int_feature(inputs['input_type_ids']),
                'label_ids': create_int_feature(label_ids)
            })).SerializeToString()
        )
    
    # sentence token in each batch
    record_writer.close()
    return batch_tokens, batch_labels, batch_inputs


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        
    }
    
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example
    
    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    
    return input_fn
