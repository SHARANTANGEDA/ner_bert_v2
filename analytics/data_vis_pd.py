from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def auto_label(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


LABELS = ["B-NAME", "B-LOC", "O", "B-ORG", "B-MISC"]

rows = pd.read_csv('../telugu_dataset/train_os.csv')

initial_labels = []

for idx, row in rows.iterrows():
    lbl_list = row['entity_type'].split(" ")
    initial_labels.extend(lbl_list)

init_data = []
for lbl in LABELS:
    init_data.append(initial_labels.count(lbl))
print(init_data)
fig, ax = plt.subplots()
x = np.arange(len(LABELS))  # the label locations
rects1 = ax.bar(LABELS, init_data, 0.35)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of units')
ax.set_title('Labels')
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend()
print(len(initial_labels))
auto_label(rects1)
fig.tight_layout()

plt.show()
