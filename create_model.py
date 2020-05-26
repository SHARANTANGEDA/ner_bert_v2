from tokenizer import preprocess_data
import tensorflow as tf
from bert import run_classifier
from bert import modeling

bert_config = modeling.BertConfig.from_json_file("./multi_cased_L-12_H-768_A-12/bert_config.json")
# model = run_classifier.create_model(bert_config=bert_config, is_training=True, input_ids=)