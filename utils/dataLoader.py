import torchtext
import os
import spacy
import re
from string import punctuation
from itertools import groupby

nlp = spacy.load("en_core_web_sm")


def tokenizer(sent): return [
    x.text for x in nlp.tokenizer(sent) if x.text != " "]


def loadDataset(dataset, batch_size, device):
    if dataset == 'AGNews':
        # load AGNews dataset
        return AGNewsLoader(
            batch_size=batch_size, device=device)
    elif dataset == 'FakeNews':
        # load FakeNews dataset
        return FakeNewsLoader(
            batch_size=batch_size, device=device)
    elif dataset == 'Liar2':
        return Liar2Loader(
            batch_size=batch_size, device=device)
    elif dataset == 'Liar6':
        return Liar6Loader(
            batch_size=batch_size, device=device)
    elif dataset == 'SemTopic':
        return SemTopic(
            batch_size=batch_size, device=device)
    elif dataset == 'SemStance':
        return SemStance(
            batch_size=batch_size, device=device)
    elif dataset == 'SemSentiment':
        return SemSentiment(
            batch_size=batch_size, device=device)
    elif dataset == 'Fake':
        return Fake(
            batch_size=batch_size, device=device)
    elif dataset == 'FakeTopic':
        return FakeTopic(
            batch_size=batch_size, device=device)
    elif dataset == 'sentiment':
        return sentiment(
            batch_size=batch_size, device=device)
    elif dataset == 'FakeNews1':
        return FakeNews1(
            batch_size=batch_size, device=device)
    elif dataset == 'FakeNews2':
        return FakeNews2(
            batch_size=batch_size, device=device)
    elif dataset == 'BDCI':
        return BDCILoader(
            batch_size, device=device)
    return None


def removeURL(string):
    return re.sub(r'http\S+', '', string)


def removeHashtag(string):
    string = re.sub(r'#\w+ ?', '', string)
    return re.sub(r'@\w+ ?', '', string)


def removeEmoji(string):
    return emoji.get_emoji_regexp().sub(u' ', string)


def removeRepeated(string):
    punc = set(list(punctuation)+['，', '。', '？', '！', ' ']) - {'.'}
    newtext = []
    for k, g in groupby(string):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return (''.join(newtext))


def replaceSpaces(string):
    parseString = re.sub('\t\r\n\f\v+', ' ', string).strip()
    return parseString.strip()


def BDCILoader(batch_size, device, dir='data/BDCI2019', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, include_lengths=True, tokenize=tokenizer)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('text', text_field), ('content', None), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('text', text_field), ('content', None), ('label', label_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def AGNewsLoader(batch_size, device, dir='data/AGNews', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    test_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                 fields=[
                                                     ('label', label_field), ('text', text_field)],
                                                 skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    test_iterator = torchtext.data.BucketIterator(
        test_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator, test_iterator


def FakeNewsLoader(batch_size, device, dir='data/FakeNews', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('text', text_field), ('content', None), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('text', text_field), ('content', None), ('label', label_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def Liar6Loader(batch_size, device, dir='data/Liar-6', format='tsv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def Liar2Loader(batch_size, device, dir='data/Liar-2', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def SemTopic(batch_size, device, dir='data/SemEval2016', format='tsv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def SemStance(batch_size, device, dir='data/SemEval2016', format='tsv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('target', None), ('text', text_field), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                  fields=[
                                                      ('id', None), ('target', None), ('text', text_field), ('label', label_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def sentiment(batch_size, device, dir='data/sentiment', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                  fields=[
                                                      ('id', None), ('label', label_field), ('text', text_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def SemSentiment(batch_size, device, dir='data/SemEval2016', format='tsv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('target', None), ('text', text_field), ('stance', None), ('other', None), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                  fields=[
                                                      ('id', None), ('target', None), ('text', text_field), ('stance', None), ('other', None), ('label', label_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def Fake(batch_size, device, dir='data/Fake', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('url', None), ('text', text_field), ('label', label_field), ('category', None)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('url', None), ('text', text_field), ('label', label_field), ('category', None)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def FakeTopic(batch_size, device, dir='data/Fake', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('url', None), ('text', text_field), ('fake', None), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('url', None), ('text', text_field), ('fake', None), ('label', label_field)],
                                                  skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator


def FakeNews1(batch_size, device, dir='data/FakeNews', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('url', None), ('title', None), ('text', text_field), ('label', label_field), ('isFake', None)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('url', None), ('title', None), ('text', text_field), ('label', label_field), ('isFake', None)],
                                                  skip_header=True)
    test_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                 fields=[
                                                     ('url', None), ('title', None), ('text', text_field), ('label', label_field), ('isFake', None)],
                                                 skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    test_iterator = torchtext.data.BucketIterator(
        test_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator, test_iterator


def FakeNews2(batch_size, device, dir='data/FakeNews', format='csv'):
    text_field = torchtext.data.Field(
        lower=True, sequential=True, tokenize=tokenizer, include_lengths=True)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('url', None), ('title', None), ('text', text_field), ('category', None), ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('url', None), ('title', None), ('text', text_field), ('category', None), ('label', label_field)],
                                                  skip_header=True)
    test_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                 fields=[
                                                     ('url', None), ('title', None), ('text', text_field), ('category', None), ('label', label_field)],
                                                 skip_header=True)
    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=True)
    test_iterator = torchtext.data.BucketIterator(
        test_dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, train_dataset, valid_dataset, train_iterator, valid_iterator, test_iterator
