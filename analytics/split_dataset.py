lines = open('../telugu_dataset/total_over_sampled.csv').readlines()

title = lines[0]
lines = lines[1:]


from sklearn.model_selection import train_test_split
train, test = train_test_split(lines, test_size=0.1)

train, val = train_test_split(train, test_size=0.1)


print(len(train), len(val), len(test))

file = open("../telugu_dataset/train_os.csv", "w+")
file.write(title)
file.writelines(train)


file = open("../telugu_dataset/validation_os.csv", "w+")
file.write(title)
file.writelines(val)


file = open("../telugu_dataset/test_os.csv", "w+")
file.write(title)
file.writelines(test)