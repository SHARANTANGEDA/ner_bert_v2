import pandas as pd

# text = open('NER_data.csv', 'r').readlines()
# t_f = []
# for line in text:
#     if line.startswith(","):
#         t_f.append('"' + line[0] + '"' + line[1:])
#     else:
#         t_f.append(line)
# file = open('NER_data.csv', 'w')
# file.writelines(t_f)

data, sentence, label = [], [], []
df = pd.read_csv("NER_data.csv")
couple = False

for idx, row in df.iterrows():
    sentence.append(row['Word'])
    label.append(row['Label'])
    if row['Word'] == "." and couple:
        label = ' '.join([lbl for lbl in label if len(lbl) > 0])
        sentence = ' '.join([word for word in sentence if len(word) > 0])
        data.append((sentence, label))
        sentence, label = [], []
        couple = False
    elif row['Word'] == '.' and not couple:
        couple = True

df_formatted = pd.DataFrame(data, columns=['sequence', 'entity_type'])
df_formatted.to_csv("telugu_dataset/train_merged.csv", header=False, index=False, mode='a')

#
# text = open('NER_data.csv', 'r').readlines()
# t_f = []
# for line in text:
#     if not line.startswith('","'):
#         t_f.append(line)
# file = open('NER_data.csv', 'w')
# file.writelines(t_f)