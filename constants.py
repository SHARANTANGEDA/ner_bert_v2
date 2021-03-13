import os

"""Initialize ENV Variables"""
PRE_TRAINED_MODEL_DIR = os.getenv("BERT_PRE_TRAINED_MODEL")
BERT_VOCAB_FILE = os.path.join(PRE_TRAINED_MODEL_DIR, "vocab.txt")
CONFIG_JSON_FILE = os.path.join(PRE_TRAINED_MODEL_DIR, "bert_config.json")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH"))
PROCESSED_DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
ML_FLOW_SAVE_DIR = os.getenv("ML_FLOW_SAVE_DIR")
LOGS_DIR = os.getenv("LOGS_DIR")
ML_FLOW_EXPERIMENT_ID = os.getenv("ML_FLOW_EXPERIMENT_ID", 0)

"""
here "X" used to represent "##eer","##soo" and so on!
"[PAD]" for padding
:return:
"""

LABELS = ["[PAD]", "B-NAME", "B-LOC", "O", "B-ORG", "B-MISC", "[CLS]", "[SEP]", "X"]
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_TENSOR_RECORD = "train.tf_record"
VAL_TENSOR_RECORD = "validation.tf_record"
TEST_TENSOR_RECORD = "test.tf_record"

MID_OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, "mid_out")
FINAL_OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, "final_out")
LABEL_ID_PKL_FILE = os.path.join(MID_OUTPUT_DIR, "label2id.pkl")
BERT_CONFIG_JSON_FILE = os.path.join(PRE_TRAINED_MODEL_DIR, "bert_config.json")
BERT_MODEL_FILE = os.path.join(PRE_TRAINED_MODEL_DIR, "bert_model.ckpt")

TENSOR_TRAIN_FEATURES_RECORD_FILE = os.path.join(MODEL_OUTPUT_DIR, TRAIN_TENSOR_RECORD)
TENSOR_VAL_FEATURES_RECORD_FILE = os.path.join(MODEL_OUTPUT_DIR, VAL_TENSOR_RECORD)
TENSOR_TEST_FEATURES_RECORD_FILE = os.path.join(MODEL_OUTPUT_DIR, TEST_TENSOR_RECORD)
