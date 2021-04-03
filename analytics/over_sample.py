import random

import pandas as pd

LABELS = ["B-NAME", "B-LOC", "O", "B-ORG", "B-MISC"]

data = pd.read_csv("../telugu_dataset/total.csv")

over_sample_tokens = []

for idx, row in data.iterrows():
    labels = row['entity_type']
    lbl_list = labels.split(" ")
    if lbl_list.count("O") + 1 < len(lbl_list) - lbl_list.count("O"):
        over_sample_tokens.append(idx)

print(len(over_sample_tokens))
cnt = len(over_sample_tokens)
sample_times = 20
# 2867
rows_std = open('../telugu_dataset/total.csv', 'r').readlines()
rows_std = rows_std[1:]

# Over Sample the Dataset
rows = open('../telugu_dataset/total.csv', 'r').readlines()
title = rows[0]
rows = rows[1:]

for row_idx in over_sample_tokens:
    for _ in range(sample_times):
        rows.insert(random.randint(0, len(rows)), rows_std[row_idx])

file = open("../telugu_dataset/total_over_sampled.csv", "w+")

file.write(title)
file.writelines(rows)
file.close()

