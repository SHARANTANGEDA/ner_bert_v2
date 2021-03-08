from bert_model.model import train_test
import mlflow
import constants as c

mlflow.tensorflow.autolog(log_models=True, disable=False, exclusive=False)
save_dir_path = train_test(epochs=3, train_batch_size=32, eval_batch_size=32, warmup_proportion=0.1, init_lr=2e-5)
mlflow.tensorflow.save_model(save_dir_path, path=c.ML_FLOW_SAVE_DIR)
