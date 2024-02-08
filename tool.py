import string
import torch, csv
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from collections import Counter


class DataLoad(Dataset):
    def __init__(self, split, data_dir="./datasets/"):
        assert split in ["train", "test"]
        self.split = split
        self.data_dir = data_dir

        self.pairs = self.load_data()

    def load_data(self):
        pairs = []
        with open(join(self.data_dir, self.split+".csv"), errors='ignore') as f:
            for line in f:
                label, sentence = line.split(",", 1)
                sentence = sentence.strip("\n")
                label = int(label) - 1
                pairs.append((label, sentence))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def load_stopwords(path=".//datasets/stopwords.txt"):
    """加载禁用词 集合"""
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        stopwords = [word.strip('\n') for word in f]
    stopwords += list(string.printable)
    return set(stopwords)


def load_word2id(length=2000, vocab_path="./datasets/vocab.csv"):
    """加载词典为字典对象：{"text1":0, "text2":1,...}"""
    word2id = {"<pad>": 0, "<unk>": 1}
    with open(vocab_path, "r", encoding='utf-8', errors='ignore') as f:
        words = [line.split(',')[0] for line in f]
    for word in words[:length]:
        word2id[word] = len(word2id)
    return word2id


def collate_fn_ml(word2id, batch):
    """功能:为ML分类方法提供数据，这里主要是将文本转化为向量"""
    labels, sentences = zip(*batch)
    labels = torch.LongTensor(labels)

    bsize = len(sentences)
    length = len(word2id)
    sent_tensor = torch.zeros(bsize, length).long()
    for sent_id, sent in enumerate(sentences):
        for gram in sent:
            if gram in word2id:
                gram_id = word2id[gram]
                sent_tensor[sent_id][gram_id] += 1

    return labels, sent_tensor


def collate_fn_dl(word2id, batch):
    """为DL分类方法提供数据：将数据集中的所有单词转化为word2id中对应的id"""
    batch.sort(key=lambda pair: len(pair[1]), reverse=True)
    labels, sentences = zip(*batch)
    sentences = [sent[:64] for sent in sentences]
    labels = torch.LongTensor(labels)
    pad_id = word2id["<pad>"]  # 0
    unk_id = word2id["<unk>"]  # 1
    bsize = len(sentences)
    max_len = len(sentences[0])
    sent_tensor = torch.ones(bsize, max_len).long() * pad_id
    for sent_id, sent in enumerate(sentences):
        for word_id, word in enumerate(sent):
            sent_tensor[sent_id][word_id] = word2id.get(word, unk_id)

    lengths = [len(sent) for sent in sentences]
    return labels, sent_tensor, lengths


def preprocess_for_ml(sentences):
    # 将字与字之间用空格隔开(分词)：将每个字符串中的字符以空格隔开，并将结果存储回原列表中
    sentences = [" ".join(list(sent)) for sent in sentences]
    return sentences


def get_features(sent):
    """抽取1-gram 以及 2-gram特征：从该字符串中提取1-gram（单个字符）和2-gram（相邻的两个字符对）特征"""
    unigrams = list(sent)
    bigrams = [sent[i:i+2] for i in range(len(sent)-1)]
    return unigrams + bigrams


class Voc(object):
    """构建N-Gram词典"""
    def __init__(self, N=1):
        self.N = N
        self.stopwords = load_stopwords()
        self.gram2id = {}
        self.id2gram = {}
        self.length = 0
        self.gram2count = {}

    def add_sentence(self, sentence):
        ngrams = [sentence[i:i+self.N] for i in range(len(sentence)-self.N-1)]
        ngrams = self.filter_stopgram(ngrams)
        for ngram in ngrams:
            self.add_gram(ngram)

    def __len__(self):
        return self.length

    def __str__(self):
        return "{}-Gram Voc(Length: {})".format(self.N, self.length)

    def __repr__(self):
        return str(self)

    def trim(self, min_count=3):
        """将出现次数少于min_count的词从字典中去掉"""
        keep_grams = []
        for gram, count in self.gram2count.items():
            if count >= min_count:
                keep_grams.append(gram)
        self.gram2id = {}
        self.id2gram = {}
        self.length = 0
        for gram in keep_grams:
            self.add_gram(gram)

    def add_gram(self, ngram):
        if ngram not in self.gram2id:
            self.gram2id[ngram] = self.length
            self.id2gram[self.length] = ngram
            self.gram2count[ngram] = 1
            self.length += 1
        else:
            self.gram2count[ngram] += 1

    def filter_stopgram(self, ngrams):
        """过滤禁用词"""
        filtered_ngrams = []
        for ngram in ngrams:
            # 与禁用词表没有交集
            if not set(ngram).intersection(self.stopwords):
                filtered_ngrams.append(ngram)

        return filtered_ngrams


def make_vocab(path, i):
    voc = Voc(N=i)
    train_dataset = DataLoad("train")
    for _, sentence in train_dataset:
        voc.add_sentence(sentence)

    counter = Counter(voc.gram2count)
    with open(path, "a", encoding='utf-8', errors='ignore') as f:
        for word, count in counter.most_common():
            f.write(word + ',' + str(count) + '\n')
    print("Build Vocab Done!")


def csv_sort(path):
    df = pd.read_csv(path, header=None, names=['col_0', 'col_1'])
    df_sorted = df.sort_values(by='col_1', ascending=False)
    df_sorted.to_csv(path, index=False, header=None)
    print("Sort completed")


def play_vocab(path):
    for i in range(1, 3):
        make_vocab(path, i)
    csv_sort(path)