import logging
import os
import pickle
from datetime import datetime

import constants as c
from metrics.metrics import recall_m, precision_m, f1_m
from ner_utils import pre_process, extract_features
from tensorflow import keras
import tensorflow as tf

from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.tokenization
import numpy as np


def train_test(epochs, train_batch_size, eval_batch_size, warmup_proportion=0.1, init_lr=2e-5):
    """Create Features & Tokenize"""
    logging.getLogger().setLevel(logging.INFO)
    tokenizer = bert.tokenization.FullTokenizer(c.BERT_VOCAB_FILE, do_lower_case=True)
    logging.info(f'Vocab Size: {tokenizer.vocab}')
    train_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.TRAIN_FILE), c.TRAIN_FILE)
    test_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.TEST_FILE), c.TEST_FILE)
    val_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.VALIDATION_FILE), c.VALIDATION_FILE)
    
    train_data_size = len(train_data)
    num_train_steps = int(train_data_size / train_batch_size * epochs)
    warmup_steps = int(epochs * warmup_proportion * train_data_size / train_batch_size)
    
    _, batch_train_labels, batch_train_inputs = extract_features.convert_data_into_features(
        train_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_TRAIN_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    _, batch_val_labels, batch_val_inputs = extract_features.convert_data_into_features(
        val_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_VAL_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    _, batch_test_labels, batch_test_inputs = extract_features.convert_data_into_features(
        test_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_TEST_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    
    # Initialize BERT Model
    bert_config = bert.configs.BertConfig.from_json_file(c.BERT_CONFIG_JSON_FILE)
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=len(c.LABELS),
                                                                      max_seq_length=c.MAX_SEQ_LENGTH)
    
    # Plot Model Image
    keras.utils.plot_model(bert_classifier, show_shapes=True, show_layer_names=True, dpi=50,
                           to_file='model_classifier.png')
    
    # Plot BERT Encoder
    keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=50, to_file='model_encoder.png')
    
    # Restore from last checkpoint
    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(c.BERT_MODEL_FILE).assert_consumed()
    
    # Add Adam Optimizer
    optimizer = nlp.optimization.create_optimizer(init_lr, num_train_steps=num_train_steps,
                                                  num_warmup_steps=warmup_steps, optimizer_type='adamw')
    
    # Train the model
    metrics = [keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32), recall_m, precision_m, f1_m]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    bert_classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    
    bert_classifier.fit(
        batch_train_inputs, batch_train_labels,
        validation_data=(batch_val_inputs, batch_val_labels),
        batch_size=train_batch_size,
        epochs=epochs)
    
    # Save Model
    save_dir_path = os.path.join(c.FINAL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    tf.saved_model.save(bert_classifier, export_dir=save_dir_path)
    
    # Test Scores
    test_loss, test_acc, test_recall, test_precision, test_f_score = bert_classifier.evaluate(batch_test_inputs,
                                                                                              batch_test_labels,
                                                                                              batch_size=eval_batch_size)
    logging.info("****TEST METRICS****")
    logging.info(f'Test Loss: {test_loss}')
    logging.info(f'Test Accuracy: {test_acc}')
    logging.info(f'Test Recall: {test_recall}')
    logging.info(f'Test Precision: {test_precision}')
    logging.info(f'Test F1_Score: {test_f_score}')
    return save_dir_path


def serve_with_saved_model(formatted_data, saved_classifier):
    result = saved_classifier(formatted_data, training=False)
    formatted_result = tf.argmax(result).numpy()
    label2id_map = pickle.load(open(c.LABEL_ID_PKL_FILE, "r"))
    id2label_map = {v: k for k, v in label2id_map.items()}
    return np.vectorize(id2label_map.get)[formatted_result]
