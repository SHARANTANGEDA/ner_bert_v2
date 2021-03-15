import logging
import os
import pickle
import uuid
from datetime import datetime

import mlflow
from sklearn.metrics import classification_report, f1_score
from transformers import BertConfig, TFBertForTokenClassification, BertTokenizer
from tensorflow import keras
import tensorflow as tf
import numpy as np

from metrics.metrics import f1_m, recall_m, precision_m
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
    
    metrics = [keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32), f1_m, recall_m, precision_m]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    logging.info("Compiling Model ...")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    logging.info("Model has been compiled")
    
    val_data, val_inputs, val_labels = extract_features.retrieve_features(c.VALIDATION_FILE, c.LABELS, c.MAX_SEQ_LENGTH,
                                                                          tokenizer, c.LABEL_ID_PKL_FILE)
    test_data, test_inputs, test_labels = extract_features.retrieve_features(c.TEST_FILE, c.LABELS, c.MAX_SEQ_LENGTH,
                                                                             tokenizer, c.LABEL_ID_PKL_FILE)
    
    logging.info("Test Validation features are ready")
    
    # f1_metric = F1Metric(val_data, val_labels)
    
    model.fit(train_data, epochs=epochs, validation_data=val_data)
    
    logging.info("Model Fitting is done")
    
    # Save Model
    save_dir_path = os.path.join(c.FINAL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    tf.saved_model.save(model, export_dir=save_dir_path)
    logging.info("Model Saved")
    
    # Test Scores
    test_loss, test_acc = model.evaluate(test_data, batch_size=eval_batch_size)
    logging.info({"Loss": test_loss, "Accuracy": test_acc})
    
    # evaluate model with sklearn
    predictions = model.predict(test_data, batch_size=eval_batch_size, verbose=1).to_tuple()
    print(predictions)
    preds = np.asarray(predictions[1], dtype=np.float).round()
    sk_report = classification_report(test_labels, preds, digits=len(c.LABELS), labels=c.LABELS)
    f1_score_sk = f1_score(test_labels, preds, labels=c.LABELS, average='micro')
    
    print('\n')
    print(sk_report)
    logging.info(sk_report)
    
    logging.info("****TEST METRICS****")
    metrics_dict = {"Loss": test_loss, "Accuracy": test_acc, "Micro_F_Score": f1_score_sk}
    logging.info(str(metrics_dict))
    mlflow.log_metrics(metrics_dict)
    
    return save_dir_path, [
        f'epochs:{epochs}', f'eval_batch_size: {eval_batch_size}', f'epsilon: {epsilon}', f'init_lr: {init_lr}',
        f'beta_1: {beta_1}', f'beta_2: {beta_2}'], f'bert_{test_acc}_{f1_score_sk}_{uuid.uuid4()}'


def serve_with_saved_model(formatted_data, saved_classifier):
    result = saved_classifier(formatted_data, training=False)
    formatted_result = tf.argmax(result).numpy()
    label2id_map = pickle.load(open(c.LABEL_ID_PKL_FILE, "r"))
    id2label_map = {v: k for k, v in label2id_map.items()}
    return np.vectorize(id2label_map.get)[formatted_result]
