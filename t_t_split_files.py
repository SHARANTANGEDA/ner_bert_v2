f = open('labelled_ner.txt', 'r').readlines()
train = open('train.txt', 'w+')
test = open('test.txt', 'w+')

for idx, line in enumerate(f):
	if idx < 600001:
		train.write(line)
	else:
		test.write(line)