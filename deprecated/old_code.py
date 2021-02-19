import logging
import os

import constants as c
from metrics.metrics import recall_m, precision_m, f1_m
from ner_utils import pre_process, extract_features

from tensorflow import keras
from official.nlp import bert

# Load the required submodules
import official.nlp.bert.tokenization

# TODO: This code needs bert-for-tf2 library this will not work with current libraries

def train_test(train_batch_size, train_epochs, warmup_proportion):
    """Create Features & Tokenize"""
    logging.getLogger().setLevel(logging.INFO)
    tokenizer = bert.bert_tokenization.FullTokenizer(c.BERT_VOCAB_FILE, do_lower_case=True)
    train_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.TRAIN_FILE), c.TRAIN_FILE)
    
    num_train_steps = int(len(train_data) / train_batch_size * train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    extract_features.convert_data_into_features(
        train_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, os.path.join(c.MODEL_OUTPUT_DIR, c.TRAIN_TENSOR_RECORD),
        c.LABEL_ID_PKL_FILE)
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_data))
    logging.info("  Batch size = %d", train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)


# INCOMPLETE
def model_fn_builder(num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                     max_seq_length, crf, train_features):
    logging.info("*** Features ***")
    for name in sorted(train_features.keys()):
        logging.info("  name = %s, shape = %s" % (name, train_features[name].shape))
    input_ids = train_features["input_ids"]
    mask = train_features["mask"]
    segment_ids = train_features["segment_ids"]
    label_ids = train_features["label_ids"]
    model = create_model(input_ids, mask, segment_ids, label_ids, num_labels,
                         max_seq_length, crf, learning_rate, num_warmup_steps)


def create_model(input_ids, segment_ids, labels, num_labels, learning_rate,
                 epochs, train_batch_size, val_batch_size, test_input_ids, test_segment_ids, test_labels):
    bert_params = bert.params_from_pretrained_ckpt(c.PRE_TRAINED_MODEL_DIR)
    # bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    # bert_output = bert_layer([input_ids, segment_ids])
    model = keras.Sequential(
        [
            bert.BertModelLayer.from_params(bert_params, name="bert"),
            keras.layers.Dropout(rate=0.1, name="dropout_train"),
            keras.layers.Dense(num_labels, activation=None, name="dense_classification"),
            keras.layers.Activation('softmax')
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['acc', recall_m, precision_m, f1_m])
    history = model.fit([input_ids, segment_ids], labels, batch_size=train_batch_size,
                        validation_batch_size=val_batch_size, epochs=epochs, validation_split=0.9)
    return model

