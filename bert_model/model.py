import logging
import os
import pickle
import uuid
from datetime import datetime

import mlflow
from transformers import BertConfig, TFBertForTokenClassification, BertTokenizer
from metrics.metrics import recall_m, precision_m, f1_m
from tensorflow import keras
import tensorflow as tf
import numpy as np

from ner_utils import extract_features
import constants as c


def train_test(epochs, eval_batch_size, epsilon=1e-7, init_lr=2e-5, beta_1=0.9, beta_2=0.999):
    mlflow.log_params({"epochs": epochs, "eval_batch_size": eval_batch_size, "epsilon": epsilon,
                       "init_lr": init_lr, "beta_1": beta_1, "beta_2": beta_2})

    """Create Features & Tokenize"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    train_data = extract_features.retrieve_features(c.TRAIN_FILE, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer,
                                                    c.LABEL_ID_PKL_FILE)
    
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=len(c.LABELS))
    model = TFBertForTokenClassification.from_pretrained("bert-base-multilingual-cased", config=config)
    model.summary()
    
    model.layers[-1].activation = tf.keras.activations.softmax
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
    
    metrics = [f1_m, keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32), recall_m, precision_m]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    logging.info("Compiling Model ...")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    logging.info("Model has been compiled")
    
    val_data = extract_features.retrieve_features(c.VALIDATION_FILE, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer,
                                                  c.LABEL_ID_PKL_FILE)
    test_data = extract_features.retrieve_features(c.TEST_FILE, c.LABELS, c.MAX_SEQ_LENGTH, tokenizer,
                                                   c.LABEL_ID_PKL_FILE)
    
    logging.info("Test Validation features are ready")
    
    model.fit(train_data, epochs=epochs, validation_data=val_data)
    
    logging.info("Model Fitting is done")
    
    # Save Model
    save_dir_path = os.path.join(c.FINAL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    tf.saved_model.save(model, export_dir=save_dir_path)
    logging.info("Model Saved")
    
    # Test Scores
    test_loss, test_acc, test_recall, test_precision, test_f_score = model.evaluate(test_data,
                                                                                    batch_size=eval_batch_size)
    
    metrics_dict = {"Loss": test_loss, "Accuracy": test_acc, "Recall": test_recall, "Precision": test_precision,
                    "F1_Score": test_f_score}
    logging.info("****TEST METRICS****")
    logging.info(str(metrics_dict))
    mlflow.log_metrics(metrics_dict)

    return save_dir_path, [
        f'epochs:{epochs}', f'eval_batch_size: {eval_batch_size}', f'epsilon: {epsilon}', f'init_lr: {init_lr}',
        f'beta_1: {beta_1}', f'beta_2: {beta_2}'], f'bert_{test_acc}_{test_f_score}_{uuid.uuid4()}'


def serve_with_saved_model(formatted_data, saved_classifier):
    result = saved_classifier(formatted_data, training=False)
    formatted_result = tf.argmax(result).numpy()
    label2id_map = pickle.load(open(c.LABEL_ID_PKL_FILE, "r"))
    id2label_map = {v: k for k, v in label2id_map.items()}
    return np.vectorize(id2label_map.get)[formatted_result]
