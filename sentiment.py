import fasttext

model = fasttext.train_supervised(
    input="train.txt", epoch=100, lr=1.5, wordNgrams=3, verbose=2, minCount=1)
print(model.test("valid.txt"))
