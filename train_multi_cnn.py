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

from utils.dataLoader import loadDataset
import utils
import model.net as net
from sklearn.metrics import f1_score
import random

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', default='data/glove.6B.200d.txt',
                    help="Path of the pretrained word embedding file")
parser.add_argument('--embedding_size', default=200,
                    help="Dimention of the word embedding layer")
parser.add_argument('--batch_size', default=32,
                    help="Batch size for training")
parser.add_argument('--dataset1', default='FakeNews1',
                    help="Dataset for Task 1")
parser.add_argument('--dataset2', default='FakeNews2',
                    help="Dataset for Task 2")
parser.add_argument('--number_of_epoches', default=50,
                    help="Number of epoches for training")
parser.add_argument('--learning_rate', default=0.001,
                    help="Learning rate for backpropagation")
parser.add_argument('--weight_decay', default=1e-5,
                    help="Weight Decay to avoid overfitting")
parser.add_argument('--output_dir', default='./experiments/base_model',
                    help="Weight Decay to avoid overfitting")
parser.add_argument('--random_seed', default=42,
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
    if len(truth):
        return right / len(truth)
    else:
        return 0


def train_epoch_progress(model, train_iterator1, train_iterator2, loss_function, optimizer, text_field1, text_field2, label_field1, label_field2, number_of_epoches, device):
    model.train()
    avg_loss = 0.0
    truth_res1 = []
    pred_res1 = []

    truth_res2 = []
    pred_res2 = []
    count = 0
    stop = int(len(train_iterator2))
    for batch2 in tqdm(train_iterator2, desc='Train epoch ' + str(epoch+1)):
        if epoch >= 0:
            batch = next(iter(train_iterator1))
            (text, text_size), label = batch.text, batch.label
            text.to(device)
            label.to(device)
            label.data.sub_(1)
            truth_res1 += list(label.data)
            for param in model.convs1.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True
            pred = model(text.t(), '1')
            if device == 'cuda':
                pred_label = pred.data.max(1)[1].cpu().numpy()
            else:
                pred_label = pred.data.max(1)[1].numpy()
            pred_res1 += [x for x in pred_label]
            loss1 = loss_function(pred, label)
            #loss = loss1
            # model.zero_grad()
            # loss1.backward(retain_graph=True)
            # optimizer.step()
        #batch2 = next(iter(train_iterator2))
        if epoch >= 0:
            (text2, text_size2), label2 = batch2.text, batch2.label
            text2.to(device)
            label2.to(device)
            label2.data.sub_(1)
            truth_res2 += list(label2.data)
            for param in model.convs1.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True
            pred2 = model(text2.t(), '2')
            if device == 'cuda':
                pred_label = pred2.data.max(1)[1].cpu().numpy()
            else:
                pred_label = pred2.data.max(1)[1].numpy()
            pred_res2 += [x for x in pred_label]
            loss2 = loss_function(pred2, label2)
            model.zero_grad()
            loss = loss1 + loss2
            loss.backward(retain_graph=True)
            optimizer.step()
        avg_loss += loss.item()

    avg_loss /= len(train_iterator1)
    acc1 = get_accuracy(truth_res1, pred_res1)
    acc2 = get_accuracy(truth_res2, pred_res2)
    return avg_loss, acc1, acc2


def valid_epoch(model, valid_iterator, loss_function, USE_GPU, task):
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
        # model.batch_size = len(label.data)
        # model.hidden1 = model.init_hidden1()
        # model.hidden2 = model.init_hidden2()
        # pred, _ = model(text, text_size, task)
        pred = model(text.t(), task)
        # pred = model(text, task)
        if device == 'cuda':
            pred_label = pred.data.max(1)[1].cpu().numpy()
        else:
            pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(valid_iterator)
    acc = get_accuracy(truth_res, pred_res)
    f1_macro1 = f1_score(truth_res, pred_res, average='macro', pos_label=1)
    f1_micro1 = f1_score(truth_res, pred_res, average='micro', pos_label=1)
    f1_macro2 = f1_score(truth_res, pred_res, average='macro', pos_label=0)
    f1_micro2 = f1_score(truth_res, pred_res, average='micro', pos_label=0)
    f1_macro = (f1_macro1 + f1_macro2)/2
    f1_micro = (f1_micro1 + f1_micro2)/2
    return avg_loss, acc, f1_macro, f1_micro


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    embedding_path = args.embedding_path
    embedding_size = args.embedding_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    random_seed = args.random_seed
    number_of_epoches = args.number_of_epoches
    dataset1 = args.dataset1
    dataset2 = args.dataset2
    output_dir = args.output_dir
    output_path = output_dir + '/' + 'Mixed'
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

    # Set the random seed for reproducible experiments
    random.seed(random_seed)
    # Set the random seed for reproducible experiments
    torch.manual_seed(random_seed)
    if isCuda:
        torch.cuda.manual_seed(random_seed)

    else:
        raise Exception
    text_field1, label_field1, train_dataset1, valid_dataset1, train_iterator1, valid_iterator1 = loadDataset(
        dataset1, batch_size, device)
    text_field2, label_field2, train_dataset2, valid_dataset2, train_iterator2, valid_iterator2 = loadDataset(
        dataset2, batch_size, device)

    # Build Vocabulary
    vec = vocab.Vectors(embedding_path)

    text_field1.build_vocab(train_dataset1, valid_dataset1, train_dataset2, valid_dataset2,
                            max_size=35000, min_freq=2, vectors=vec)
    text_field2.build_vocab(train_dataset1, valid_dataset1, train_dataset2, valid_dataset2,
                            max_size=35000, min_freq=2, vectors=vec)
    label_field1.build_vocab(train_dataset1, valid_dataset1)
    label_field2.build_vocab(train_dataset2, valid_dataset2)

    vocab_size1 = len(text_field1.vocab)
    label_size1 = len(label_field1.vocab) - 1
    vocab_size2 = len(text_field2.vocab)
    label_size2 = len(label_field2.vocab) - 1

    with open(os.path.join(output_path, "text_field1.field"), "wb")as f:
        dill.dump(text_field1, f)

    with open(os.path.join(output_path, "label_field1.field"), "wb")as f:
        dill.dump(label_field1, f)

    with open(os.path.join(output_path, "text_field2.field"), "wb")as f:
        dill.dump(text_field2, f)

    with open(os.path.join(output_path, "label_field2.field"), "wb")as f:
        dill.dump(label_field2, f)

    # print text and label frequecies
    print('Number of Vocabulary 1: ', vocab_size1)
    print(label_field1.vocab.freqs)
    print('Number of Label 1: ', label_size1)

    print('Number of Vocabulary 2: ', vocab_size2)
    print(label_field2.vocab.freqs)
    print('Number of Label 2: ', label_size2)

    # create model

    """
    model = net.textMultiLSTM2(vectors=text_field1.vocab.vectors, embedding_size=embedding_size, hidden_size=64,
                               batch_size=batch_size, vocab_size=vocab_size1, label_size1=label_size1, label_size2=label_size2, num_layers=2, device=device)

    model = net.MultiAttnClassifier(vectors=text_field1.vocab.vectors, embedding_size=embedding_size, hidden_size=128,
                                    vocab_size=vocab_size1, label_size1=label_size1, label_size2=label_size2, num_layers=2, device=device)
    """
    model = net.CNNMultiText2(vocab_size1, embedding_size, label_size1=label_size1, label_size2=label_size2, kernel_num=100,
                              dropout=0.5, vectors=text_field1.vocab.vectors)
    """
    model = net.LSTMCNNMultiText(vocab_size1, embedding_size, label_size1,
                                 label_size2, batch_size, vectors=text_field1.vocab.vectors)
    """
    model.to(device)

    # define the loss function and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # loss_function = nn.NLLLoss()
    loss_function = nn.CrossEntropyLoss()
    # start training
    high_acc_1 = 0
    high_acc_2 = 0
    for epoch in range(number_of_epoches):
        avg_loss, acc1, acc2 = train_epoch_progress(
            model, train_iterator1, train_iterator2, loss_function, optimizer, text_field1, text_field2, label_field1, label_field2, number_of_epoches, device)
        train_log = 'Epoch: %s Train: loss %.2f acc %.2f' % (
            epoch+1, avg_loss, acc1*100)
        tqdm.write(train_log)
        add_log(os.path.join(output_path, 'train.log'), train_log + '\n')

        train_log = 'Epoch: %s Train: loss %.2f acc %.2f' % (
            epoch+1, avg_loss, acc2*100)
        tqdm.write(train_log)
        add_log(os.path.join(output_path, 'train.log'), train_log + '\n')

        avg_loss, acc, f1_macro, f1_micro = valid_epoch(
            model, valid_iterator1, loss_function, device, '1')
        valid_log = 'Task 1 Epoch   %s Valid: loss %.2f acc %.2f f1Macro: %.4f f1Micro: %.4f' % (
            epoch+1, avg_loss, acc*100, f1_macro, f1_micro)
        tqdm.write(valid_log)
        if acc > high_acc_1:
            high_acc_1 = acc
        print('Valid Task 1', high_acc_1)
        add_log(os.path.join(output_path, 'train.log'), valid_log + '\n')

        avg_loss, acc, f1_macro, f1_micro = valid_epoch(
            model, valid_iterator2, loss_function, device, '2')
        valid_log = 'Task 2 Epoch   %s Valid: loss %.2f acc %.2f f1Macro: %.4f f1Micro: %.4f' % (
            epoch+1, avg_loss, acc*100, f1_macro, f1_micro)
        tqdm.write(valid_log)
        if acc > high_acc_2:
            high_acc_2 = acc
        print('Valid Task 2', high_acc_2)
        add_log(os.path.join(output_path, 'train.log'), valid_log + '\n')
        torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
