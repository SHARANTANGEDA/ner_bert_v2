import pandas as pd
from official.nlp import bert

# Load the required submodules
import official.nlp.bert.tokenization


class InputItem(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


def _create_example(df, set_type):
    examples = []
    for (idx, line) in df.iterrows():
        guid = "%s-%s" % (set_type, idx)
        texts = bert.bert_tokenization.convert_to_unicode(line[0])
        labels = bert.bert_tokenization.convert_to_unicode(line[1])
        examples.append(InputItem(guid=guid, text=texts, label=labels))
    return examples


def get_input_list(file_path, set_type):
    return _create_example(pd.read_csv(file_path), set_type)
