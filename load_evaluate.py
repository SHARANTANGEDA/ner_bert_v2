import logging
import os
from datetime import datetime

from bert_model.model import load_saved_model_test
import constants as c

logging.basicConfig(filename=os.path.join(c.LOGS_DIR, f'{datetime.now()}.txt'),
                    filemode='w+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

signature = load_saved_model_test(epochs=3, eval_batch_size=32, beta_1=0.9, beta_2=0.999, init_lr=2e-5, epsilon=1e-7)
print(signature)

