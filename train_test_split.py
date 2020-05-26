import csv
import os
import numpy as np
from tokenizer import createTokenizer

def loadData(tokenizer):
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    data_file = open("labelled_ner.txt", "r")
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    for row in data_file.readlines():
        row = row.split('\t')
        row[2] = row[2][:-1]
        print(row)
    # shuffled_set = random.sample(data, len(data))
    # training_set = shuffled_set[0:]
    # shuffled_set_test = random.sample(data_test, len(data_test))
    # testing_set = shuffled_set_test[0:]
    #
    # for el in training_set:
    #     train_set.append(el[1])
    #     zeros = [0] * classes
    #     zeros[int(el[0]) - 1] = 1
    #     train_labels.append(zeros)
    #
    # for el in testing_set:
    #     test_set.append(el[1])
    #     zeros = [0] * classes
    #     zeros[int(el[0]) - 1] = 1
    #     test_labels.append(zeros)
    
    # defineTokenizerConfig(train_set)
    
    train_tokens = map(tokenizer.tokenize, train_set)
    train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    
    train_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), train_token_ids)
    train_token_ids = np.array(list(train_token_ids))
    
    test_tokens = map(tokenizer.tokenize, test_set)
    test_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], test_tokens)
    test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
    
    test_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_token_ids)
    test_token_ids = np.array(list(test_token_ids))
    
    train_labels_final = np.array(train_labels)
    test_labels_final = np.array(test_labels)
    
    return train_token_ids, train_labels_final, test_token_ids, test_labels_final


train_set, train_labels, test_set, test_labels = loadData(createTokenizer())
