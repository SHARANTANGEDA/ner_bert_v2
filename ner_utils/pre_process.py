import pandas as pd


def _create_example(df):
    sentences, labels = [], []
    for (idx, line) in df.iterrows():
        sentences.append(line[0])
        labels.append(line[1])
    return sentences, labels


def read_dataset(file_path):
    return _create_example(pd.read_csv(file_path))


def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids}, y
