from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

import constants as c


# class F1Metric(Callback):
#
#     def __init__(self, val_data, labels):
#         super().__init__()
#         self.validation_data = val_data
#         self.label_data = labels
#
#     def on_train_begin(self, logs={}):
#         print(self.validation_data)
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         output = self.model.predict(self.validation_data, batch_size=32).to_tuple()
#         print(output)
#         preds = np.asarray(output[1], dtype=np.float)
#         print(preds.shape)
#         val_predict = preds.round()
#         val_targ = self.label_data
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
#     #
#     # def on_batch_end(self, batch, logs={}):
#     #     output = self.model.predict(self.validation_data, batch_size=32).to_tuple()
#     #     print(output[0])
#     #     preds = np.asarray(output[1], dtype=np.float)
#     #     print(preds.shape)
#     #     val_predict = preds.round()
#     #     val_targ = self.label_data
#     #     _val_f1 = f1_score(val_targ, val_predict)
#     #     _val_recall = recall_score(val_targ, val_predict)
#     #     _val_precision = precision_score(val_targ, val_predict)
#     #     self.val_f1s.append(_val_f1)
#     #     self.val_recalls.append(_val_recall)
#     #     self.val_precisions.append(_val_precision)
#     #     print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))

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
    report = classification_report(y_true_filter.numpy(), y_pred_filter.numpy(), labels=c.LABELS, output_dict=True)
    print(report)
    return str(report)


def calculate_pred_metrics(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    true_f, pred_f = y_true_filter.numpy(), y_pred_filter.numpy().reshape(len(y_true_filter), c.MAX_SEQ_LENGTH)
    return classification_report(true_f, pred_f, digits=len(c.LABELS), labels=c.LABELS), f1_score(
        true_f, pred_f, average='macro'), f1_score(true_f, pred_f, average='micro'), recall_score(
        true_f, pred_f, average='macro'), precision_score(true_f, pred_f, average='macro')
