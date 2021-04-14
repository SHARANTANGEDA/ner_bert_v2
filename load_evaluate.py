import argparse
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

parser = argparse.ArgumentParser(description='Evaluate saved model')
parser.add_argument('--model_path', type=str, dest="model_path", help="Add Absolute Path for saved model",
                    required=True)
parser.add_argument('--file_path', type=str, dest="file_path", help="Add Absolute Path for prediction file",
                    required=True)

args = parser.parse_args()
signature = load_saved_model_test(eval_batch_size=32, model_path=args.model_path, file_path=args.file_path)
