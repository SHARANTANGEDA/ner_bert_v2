import logging
from tf_version.utils import *
import tensorflow as tf
from tf_version.metrics import *
import os

# epochs = 3, max_seq_len = 75, lr =  5e-4, batch = 32
def main(do_train, do_eval, do_predict, crf, bert_config_file, vocab_file, output_dir, data_dir, middle_output,
         init_checkpoint, num_train_epochs=3,  max_seq_length=75, learning_rate=5e-4, train_batch_size=128,
         eval_batch_size=32, warmup_proportion=0.1, save_checkpoints_steps=1000, iterations_per_loop=1000,
         predict_batch_size=8):
    logging.set_verbosity(logging.INFO)
    # processors = {"ner": NerProcessor}
    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))
    # task_name = task_name.lower()
    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % (task_name))
    processor = NerProcessor()
    
    label_list = processor.get_labels()
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    tpu_cluster_resolver = None
    
    # TODO Change if to run on TPU or CoLab
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,  # [Optional] TensorFlow master URL.
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,  # How often to save model
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=8,  # Num of TPU Cores change if using TPU
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        
        num_train_steps = int(
            len(train_examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,  # if downloaded cased checkpoint you should use "False",if uncased use  "True"
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False,  # Same value as Use_TPU can be used here
        max_seq_length=max_seq_length,
        crf=crf)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)
    
    if do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        _, _ = filed_based_convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer, train_file, middle_output)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True)
        print('TRAIN FUNCTION DONE')
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('ESTIMATOR FUNCTION DONE')
    if do_eval:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_file = os.path.join(output_dir, "eval.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer, eval_file, middle_output)
        
        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as wf:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p, r, f = calculate(confusion_matrix, len(label_list) - 1)
            logging.info("***********************************************")
            logging.info("********************P = %s*********************", str(p))
            logging.info("********************R = %s*********************", str(r))
            logging.info("********************F = %s*********************", str(f))
            logging.info("***********************************************")
    
    if do_predict:
        with open(middle_output + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        
        predict_examples = processor.get_test_examples(data_dir)
        
        predict_file = os.path.join(output_dir, "predict.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                              max_seq_length, tokenizer,
                                                                              predict_file, middle_output)
        
        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", predict_batch_size)
        
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)
        
        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(output_dir, "label_test.txt")
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label, crf)


current_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12")
vocab_file = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12", "vocab.txt")
config_json_file = os.path.join(current_dir, "multi_cased_L-12_H-768_A-12", "bert_config.json")
middle_output = os.path.join(current_dir, "final_out_all_ner")
output_dir = os.path.join(current_dir, "mid_out_all_ner")
checkpoint_file = os.path.join(config_dir, "bert_model.ckpt")
main(False, True , False, True, config_json_file, vocab_file, output_dir, current_dir, middle_output, checkpoint_file)
