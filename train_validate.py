from bert_model.model import train_test

train_test(epochs=3, train_batch_size=32, eval_batch_size=32, warmup_proportion=0.1, init_lr=2e-5)
