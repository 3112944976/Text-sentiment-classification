from collections import defaultdict
import os, re, jieba, codecs, csv
import pandas as pd

# 生成stopword表
stopwords = set()
fr = open('./情感极性词典/停用词.txt', 'r', encoding='utf-8')
for word in fr:
    stopwords.add(word.strip())
# 读取否定词文件
not_word_file = open('./情感极性词典/否定词.txt', 'r+', encoding='utf-8')
not_word_list = not_word_file.readlines()
not_word_list = [w.strip() for w in not_word_list]
# 读取程度副词文件
degree_file = open('./情感极性词典/程度副词.txt', 'r+', encoding='utf-8')
degree_list = degree_file.readlines()
degree_list = [item.split(',')[0] for item in degree_list]
# 生成新的停用词表
with open('./情感极性词典/stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords:
        if (word not in not_word_list) and (word not in degree_list):
            f.write(word + '\n')


# jieba分词后去除停用词
def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('./情感极性词典/stopwords.txt', 'r', encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, seg_result))


# 找出文本中的情感词、否定词和程度副词
def classify_words(word_list):
    sen_file = open('./情感极性词典/BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    sen_list = sen_file.readlines()
    sen_dict = defaultdict()
    for i in sen_list:
        if len(i.split(' ')) == 2:
            sen_dict[i.split(' ')[0]] = i.split(' ')[1]
    not_word_file = open('./情感极性词典/否定词.txt', 'r+', encoding='utf-8')
    not_word_list = not_word_file.readlines()
    degree_file = open('./情感极性词典/程度副词.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    degree_dict = defaultdict()
    for i in degree_list:
        degree_dict[i.split(',')[0]] = i.split(',')[1]

    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            sen_word[i] = sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            not_word[i] = -1
        elif word in degree_dict.keys():
            degree_word[i] = degree_dict[word]


    sen_file.close()
    not_word_file.close()
    degree_file.close()
    return sen_word, not_word, degree_word


# 计算情感词的分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    W = 1
    score = 0
    sentiment_index = -1
    sentiment_index_list = list(sen_word.keys())
    for i in range(0, len(seg_result)):
        if i in sen_word.keys():
            score += W * float(sen_word[i])
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score


# 计算得分
def sentiment_score(sentence):
    seg_list = seg_word(sentence)
    # 将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word, not_word, degree_word = classify_words(seg_list)
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    if score <= 0:
        score = 1
    else:
        score = 2
    return score


def file_sentiment_score(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        with open('../datasets/new_comment.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for row in reader:
                processed_str = sentiment_score(row[1])
                writer.writerow([processed_str] + row[1:])
    file.close()
    output_file.close()


if __name__ == '__main__':
    file_sentiment_score('../datasets/comment.csv')