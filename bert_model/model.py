import logging
import os
import uuid
from datetime import datetime

import mlflow
from transformers import BertConfig, TFBertForTokenClassification, BertTokenizer
from tensorflow import keras
import tensorflow as tf

from metrics.metrics import macro_f1, calculate_pred_metrics, EvalMetrics
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
    
    metrics = [keras.metrics.SparseCategoricalAccuracy('micro_f1/cat_accuracy', dtype=tf.float32), macro_f1]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    logging.info("Compiling Model ...")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)
    
    logging.info("Model has been compiled")
    
    val_data, val_inputs, val_labels = extract_features.retrieve_features(c.VALIDATION_FILE, c.LABELS, c.MAX_SEQ_LENGTH,
                                                                          tokenizer, c.LABEL_ID_PKL_FILE)
    test_data, test_inputs, test_labels = extract_features.retrieve_features(c.TEST_FILE, c.LABELS, c.MAX_SEQ_LENGTH,
                                                                             tokenizer, c.LABEL_ID_PKL_FILE)
    
    logging.info("Test Validation features are ready")
    
    f1_metric = EvalMetrics(val_data, val_labels, eval_batch_size)
    model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[f1_metric])
    
    logging.info("Model Fitting is done")
    
    # Save Model
    save_dir_path = os.path.join(c.FINAL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    # tf.saved_model.save(model, export_dir=save_dir_path)
    model.save_pretrained(save_dir_path, saved_model=True)
    logging.info("Model Saved at: {}".format(save_dir_path))
    
    # Test Scores
    test_loss, test_acc, test_f1_macro = model.evaluate(test_data, batch_size=eval_batch_size)
    logging.info(str({"Loss": test_loss, "Micro F1/Accuracy": test_acc, "Macro F1": test_f1_macro}))
    
    # evaluate model with sklearn
    predictions = model.predict(test_data, batch_size=eval_batch_size, verbose=1).logits
    print(predictions)
    sk_report, macro_f1_score, micro_f1_score, macro_recall_score, macro_precision_score = calculate_pred_metrics(
        test_labels, predictions)
    
    print('\n', sk_report)
    logging.info(sk_report)
    
    logging.info("****TEST METRICS****")
    metrics_dict = {"Loss": test_loss, "CatAcc": test_acc, "Macro_F1": macro_f1_score, "Micro_F1": micro_f1_score,
                    "Macro_Precision": macro_precision_score, "Macro_Recall": macro_recall_score}
    logging.info(str(metrics_dict))
    mlflow.log_metrics(metrics_dict)
    
    return save_dir_path, [
        f'epochs:{epochs}', f'eval_batch_size: {eval_batch_size}', f'epsilon: {epsilon}', f'init_lr: {init_lr}',
        f'beta_1: {beta_1}', f'beta_2: {beta_2}'], f'bert_{test_acc}_{macro_f1_score}_{uuid.uuid4()}'


def load_saved_model_test(eval_batch_size=32, model_path="96_64"):
    """Create Features & Tokenize"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    trained_model = TFBertForTokenClassification.from_pretrained(model_path, do_lower_case=True)
    
    test_data, test_inputs, test_labels = extract_features.retrieve_features(c.TEST_FILE, c.LABELS, c.MAX_SEQ_LENGTH,
                                                                             tokenizer, c.LABEL_ID_PKL_FILE)
    # Test Scores
    test_loss, test_acc, test_f1_macro = trained_model.evaluate(test_data, batch_size=eval_batch_size)
    logging.info(str({"Loss": test_loss, "Micro F1/Accuracy": test_acc, "Macro F1": test_f1_macro}))

    # evaluate model with sklearn
    predictions = trained_model.predict(test_data, batch_size=eval_batch_size, verbose=1).logits
    print(predictions)
    sk_report, macro_f1_score, micro_f1_score, macro_recall_score, macro_precision_score = calculate_pred_metrics(
        test_labels, predictions)

    print('\n', sk_report)
    logging.info(sk_report)

    logging.info("****TEST METRICS****")
    metrics_dict = {"Loss": test_loss, "CatAcc": test_acc, "Macro_F1": macro_f1_score, "Micro_F1": micro_f1_score,
                    "Macro_Precision": macro_precision_score, "Macro_Recall": macro_recall_score}
    logging.info(str(metrics_dict))
    return f'bert_{test_acc}_{macro_f1_score}_{uuid.uuid4()}'
