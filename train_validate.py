import logging
import os
from datetime import datetime

from bert_model.model import train_test
import mlflow
import constants as c

logging.basicConfig(filename=os.path.join(c.LOGS_DIR, f'{datetime.now()}.txt'),
                    filemode='w+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

experiment = mlflow.get_experiment(c.ML_FLOW_EXPERIMENT_ID)
logging.info("Name: {}".format(experiment.name))
logging.info("Experiment_id: {}".format(experiment.experiment_id))
logging.info("Artifact Location: {}".format(experiment.artifact_location))
logging.info("Tags: {}".format(experiment.tags))
logging.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

mlflow.tensorflow.autolog(every_n_iter=1, log_models=True, disable=False, exclusive=False)
with mlflow.start_run(experiment_id=c.ML_FLOW_EXPERIMENT_ID):
    save_dir_path, tags, signature = train_test(epochs=3, eval_batch_size=32, beta_1=0.9, beta_2=0.999, init_lr=2e-5)
    mlflow.tensorflow.save_model(tf_saved_model_dir=save_dir_path, tf_meta_graph_tags=tags,
                                 tf_signature_def_key=signature, path=c.ML_FLOW_SAVE_DIR)
