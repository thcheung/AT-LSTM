import argparse
import logging
import os
import dill
from tqdm import tqdm
import numpy as np
import torch
import torchtext
import torch.nn as nn
from torchtext import vocab
import random

from utils.dataLoader import loadDataset
import utils
import model.net as net
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', default='data/glove.6B.200d.txt',
                    help="Path of the pretrained word embedding file")
parser.add_argument('--embedding_size', default=200,
                    help="Dimention of the word embedding layer")
parser.add_argument('--batch_size', default=32,
                    help="Batch size for training")
parser.add_argument('--dataset', default='FakeNews2',
                    help="Batch size for training")
parser.add_argument('--number_of_epoches', default=100,
                    help="Number of epoches for training")
parser.add_argument('--learning_rate', default=0.001,
                    help="Learning rate for backpropagation")
parser.add_argument('--weight_decay', default=1e-6,
                    help="Weight Decay to avoid overfitting")
parser.add_argument('--output_dir', default='./experiments/base_model',
                    help="Weight Decay to avoid overfitting")
parser.add_argument('--random_seed', default=41,
                    help="Random seed to reproduce the experiments")


def create_log(path, statement):
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(statement)


def add_log(path, statement):
    with open(path, mode='a', encoding='utf-8') as f:
        f.write(statement)


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iterator, loss_function, optimizer, text_field, label_field, numOfEpochs, device):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iterator, desc='Train epoch ' + str(epoch+1)):
        (text, text_size), label = batch.text, batch.label
        text.to(device)
        label.to(device)
        label.data.sub_(1)
        truth_res += list(label.data)

        model.batch_size = len(label.data)
        #model.hidden = model.init_hidden()

        pred, _ = model(text, text_size)
        #pred = model(text)
        if device == 'cuda':
            pred_label = pred.data.max(1)[1].cpu().numpy()
        else:
            pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iterator)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def train_epoch(model, train_iterator, loss_function, optimizer, device):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iterator:
        (text, text_size), label = batch.text, batch.label
        text.to(device)
        label.to(device)
        label.data.sub_(1)
        truth_res += list(label.data)

        model.batch_size = len(label.data)
        #model.hidden = model.init_hidden()

        pred, _ = model(text, text_size)
        #pred = model(text)
        if device == 'cuda':
            pred_label = pred.data.max(1)[1].cpu().numpy()
        else:
            pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iterator)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def valid_epoch(model, valid_iterator, loss_function, USE_GPU):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in valid_iterator:
        (text, text_size), label = batch.text, batch.label
        text.to(device)
        label.to(device)
        label.data.sub_(1)
        if device == 'cuda':
            truth_label = label.data.cpu().numpy()
        else:
            truth_label = label.data.numpy()
        truth_res += [x for x in truth_label]

        model.batch_size = len(label.data)
        #model.hidden = model.init_hidden()

        pred, _ = model(text, text_size)
        #pred = model(text)
        if device == 'cuda':
            pred_label = pred.data.max(1)[1].cpu().numpy()
        else:
            pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(valid_iterator)
    acc = get_accuracy(truth_res, pred_res)
    f1_micro = f1_score(truth_res, pred_res, average='macro')
    precision = precision_score(truth_res, pred_res, average='macro')
    recall = recall_score(truth_res, pred_res, average='macro')
    return avg_loss, acc, f1_micro, precision, recall


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    embedding_path = args.embedding_path
    embedding_size = args.embedding_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    number_of_epoches = args.number_of_epoches
    random_seed = args.random_seed
    dataset = args.dataset
    output_dir = args.output_dir
    output_path = output_dir + '/' + dataset
    params_log = "Batch_size:   {}\nEmbedding_size:  {}\nLearning_rate: {}\nWeight_decay:   {}\n".format(
        batch_size, embedding_size, learning_rate, weight_decay)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    create_log(os.path.join(output_path, 'train.log'), params_log + '\n')
    # use GPU if available
    isCuda = torch.cuda.is_available()
    if isCuda:
        device = 'cuda'
    else:
        device = 'cpu'

    random.seed(random_seed)
    # Set the random seed for reproducible experiments
    torch.manual_seed(random_seed)
    if isCuda:
        torch.cuda.manual_seed(random_seed)

    text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator, test_iterator = loadDataset(dataset,
                                                                                                                       batch_size=batch_size, device=device)

    # Build Vocabulary
    vec = vocab.Vectors(embedding_path)
    text_field.build_vocab(train_dataset, valid_dataset,
                           max_size=35000, min_freq=5, vectors=vec)
    label_field.build_vocab(train_dataset, valid_dataset)
    vocab_size = len(text_field.vocab)
    label_size = len(label_field.vocab) - 1

    with open(os.path.join(output_path, "text_field.field"), "wb")as f:
        dill.dump(text_field, f)

    with open(os.path.join(output_path, "label_field.field"), "wb")as f:
        dill.dump(label_field, f)

    # print text and label frequecies
    print('Number of Vocabulary: ', vocab_size)
    print(label_field.vocab.freqs)
    print('Number of Label: ', label_size)

    # create model
    model = net.AttnClassifier(
        vocab_size, embedding_size, 256, label_size, vectors=text_field.vocab.vectors)
    """
    model = net.AttnClassifier(
        vocab_size, embedding_size, 100, label_size, vectors=text_field.vocab.vectors)
    
    model = net.textLSTM(vectors=text_field.vocab.vectors, embedding_size=embedding_size, hidden_size=256,
                         batch_size=batch_size, vocab_size=vocab_size, label_size=label_size, num_layers=2, device=device)
    model = net.CNN_Text(vocab_size, embedding_size, label_size,
                         dropout=0.5, vectors=text_field.vocab.vectors)
    """
    model.to(device)

    # define the loss function and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    # loss_function = nn.NLLLoss()
    loss_function = nn.CrossEntropyLoss()
    # start training
    high_acc = 0
    test_high = 0
    f1_high = 0
    pre_high = 0
    recall_high = 0
    for epoch in range(number_of_epoches):

        avg_loss, acc = train_epoch_progress(
            model, train_iterator, loss_function, optimizer, text_field, label_field, number_of_epoches, device)
        train_log = 'Epoch: %s Train: loss %.2f acc %.2f' % (
            epoch, avg_loss, acc*100)
        tqdm.write(train_log)
        add_log(os.path.join(output_path, 'train.log'), train_log + '\n')

        avg_loss, acc, f1_micro, precision, recall = valid_epoch(
            model, valid_iterator, loss_function, device)
        valid_log = 'Epoch % s Valid: loss % .2f acc % .2f' % (
            epoch+1, avg_loss, acc*100)
        tqdm.write(valid_log)
        add_log(os.path.join(output_path, 'train.log'), valid_log + '\n')

        if acc > high_acc:
            high_acc = acc
            torch.save(model.state_dict(), os.path.join(
                output_path, "model.pth"))
            avg_loss, acc, f1_micro, precision, recall = valid_epoch(
                model, test_iterator, loss_function, device)
            test_log = 'Epoch   %s Test: loss %.2f acc %.2f f1: %.2f pre: %.2f recall: %.2f' % (
                epoch+1, avg_loss, acc*100, f1_micro*100, precision*100, recall*100)
            tqdm.write(test_log)
            add_log(os.path.join(output_path, 'train.log'), test_log + '\n')
            test_high = acc*100
            f1_high = f1_micro*100
            pre_high = precision*100
            recall_high = recall*100
        print('Highest Test acc % .2f f1: % .2f pre: % .2f recall: % .2f' % (
            test_high, f1_high, pre_high, recall_high))
