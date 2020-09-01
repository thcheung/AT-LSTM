import dill
import model.net as net
import torch
import numpy as np
import math
from flask import Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# load dataset
with open("model/text_field.field", "rb")as f:
    text_field = dill.load(f)

with open("model/label_field.field", "rb")as f:
    label_field = dill.load(f)

modelPath = "model/model.pth"

embedding_size = 200
vocab_size = len(text_field.vocab)
label_size = len(label_field.vocab) - 1
text_vocab = text_field.vocab
label_vocab = label_field.vocab


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = net.CNN_Text(vocab_size, embedding_size, label_size, kernel_num=100,
                     dropout=0.5, vectors=text_field.vocab.vectors)
model.to(device)
model.load_state_dict(torch.load(modelPath))
with torch.no_grad():
    model.load_state_dict(torch.load(modelPath))
    model.to('cpu')
    model.eval()


@app.route('/predict/sentiment', methods=['POST'])
def sentiemnt():
    data = request.json
    text = data['text']
    predictString = str(text)
    text = text_field.tokenize(predictString)
    text = [text_field.vocab.stoi[x] for x in text]
    text = np.asarray(text)
    text = torch.LongTensor(text)
    text_tensor = torch.autograd.Variable(text)
    text_tensor = torch.reshape(text_tensor, (1, len(text_tensor)))
    pred = model(text_tensor)
    pred_label = torch.argmax(pred) + 1
    prob = math.exp(torch.max(pred))
    return jsonify({"sentiment": label_field.vocab.itos[pred_label]})


if __name__ == '__main__':
    app.run(port=6000)
