import logging
import os
import pickle
from datetime import datetime

from transformers import BertConfig, TFBertForTokenClassification, BertTokenizer
from metrics.metrics import recall_m, precision_m, f1_m
from tensorflow import keras
import tensorflow as tf
import numpy as np

from ner_utils import pre_process, extract_features
import constants as c


def train_test(epochs, eval_batch_size, epsilon=1e-7, init_lr=2e-5, beta_1=0.9, beta_2=0.999):
    """Create Features & Tokenize"""
    logging.getLogger().setLevel(logging.INFO)
    train_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.TRAIN_FILE), c.TRAIN_FILE)
    test_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.TEST_FILE), c.TEST_FILE)
    val_data = pre_process.get_input_list(os.path.join(c.PROCESSED_DATASET_DIR, c.VALIDATION_FILE), c.VALIDATION_FILE)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    
    _, batch_train_labels, batch_train_inputs = extract_features.convert_data_into_features(
        train_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_TRAIN_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    _, batch_val_labels, batch_val_inputs = extract_features.convert_data_into_features(
        val_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_VAL_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    _, batch_test_labels, batch_test_inputs = extract_features.convert_data_into_features(
        test_data, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer, c.TENSOR_TEST_FEATURES_RECORD_FILE, c.LABEL_ID_PKL_FILE)
    
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=len(c.LABELS))
    model = TFBertForTokenClassification.from_pretrained("bert-base-multilingual-cased", config=config)
    
    model.layers[-1].activation = tf.keras.activations.softmax
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
    
    metrics = [keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32), recall_m, precision_m, f1_m]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=50,
                           to_file='model.png')
    
    logging.info("Pre-processing and plotting is done")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(batch_train_inputs, batch_train_labels, epochs=epochs,
              validation_data=(batch_val_inputs, batch_val_labels))
    
    logging.info("Model Fitting is done")

    
    # Save Model
    save_dir_path = os.path.join(c.FINAL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    tf.saved_model.save(model, export_dir=save_dir_path)
    logging.info("Model Fitting is done")
    
    # Test Scores
    test_loss, test_acc, test_recall, test_precision, test_f_score = model.evaluate(batch_test_inputs,
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
