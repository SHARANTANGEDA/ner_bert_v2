from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

import numpy as np

import constants as c


class EvalMetrics(Callback):
    
    def __init__(self, val_data, labels, batch_size):
        super().__init__()
        self.validation_data = val_data
        self.label_data = labels
        self.batch_size = batch_size
    
    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_f1s = []
        self.val_f1s_micro = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        predictions = np.argmax(self.model.predict(self.validation_data, batch_size=self.batch_size, verbose=1).logits,
                                axis=-1)
        print(np.shape(predictions))
        sk_report, macro_f1_score, micro_f1_score, macro_recall_score, macro_precision_score = calculate_pred_metrics(
            self.label_data, predictions)
        print(sk_report)

        self.val_f1s.append(macro_f1_score)
        self.val_recalls.append(macro_recall_score)
        self.val_precisions.append(macro_precision_score)
        self.val_f1s_micro.append(micro_f1_score)
        print(f'Micro F1: {micro_f1_score}, Macro F1: {macro_f1_score}, Recall: {macro_recall_score},'
              f' Precision: {macro_precision_score}')


def _prep_predictions(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = ops.convert_to_tensor_v2_with_dispatch(y_true)
    y_pred_rank = y_pred.shape.ndims
    y_true_rank = y_true.shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            K.int_shape(y_true)) == len(K.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)
    
    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))
    return y_true, y_pred


def macro_recall(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return recall_score(y_true_filter.numpy(), y_pred_filter.numpy(), average='macro')


def macro_precision(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return precision_score(y_true_filter.numpy(), y_pred_filter.numpy(), average='macro')


def micro_f1(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return f1_score(y_true_filter.numpy(), y_pred_filter.numpy(), average='micro')


def macro_f1(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return f1_score(y_true_filter.numpy(), y_pred_filter.numpy(), average='macro')


def get_classification_report(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    print(y_true_filter.numpy(), y_pred_filter.numpy())
    report = classification_report(y_true_filter.numpy(), y_pred_filter.numpy(), labels=c.LABELS, output_dict=True)
    print(report)
    return report


def calculate_pred_metrics(y_true, y_pred):
    # true_f, pred_f = _prep_predictions(y_true, y_pred)
    true_f, pred_f = np.reshape(y_true, (len(y_true)*c.MAX_SEQ_LENGTH,)), np.reshape(y_pred,
                                                                                     (len(y_pred)*c.MAX_SEQ_LENGTH,))
    return classification_report(true_f, pred_f, labels=c.LABELS), f1_score(true_f, pred_f, average='macro'), f1_score(
        true_f, pred_f, average='micro'), recall_score(true_f, pred_f, average='macro'), precision_score(true_f, pred_f,
                                                                                                         average='macro')
