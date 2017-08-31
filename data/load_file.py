import os
import jieba
import random

corpus_path = '/data/corpus/absa/'


def get_corpus_path(path):
    return os.path.join(corpus_path, path)


def cut_sent_list(sent_list):
    return list(map(lambda x: ' '.join(jieba.cut(x)), sent_list))


def append_dict(corpus, sent_list, pola, keyword):
    for line in sent_list:
        dict0 = {
            'text': line,
            'pola': '%s\n' % pola,
            'cate': '%s\n' % keyword
        }
        corpus.append(dict0)

if __name__ == '__main__':
    corpus = []
    food_pos = cut_sent_list(open(get_corpus_path('food_pos.txt')).readlines())
    food_neu = cut_sent_list(open(get_corpus_path('food_neu.txt')).readlines())
    food_neg = cut_sent_list(open(get_corpus_path('food_neg.txt')).readlines())
    env_pos = cut_sent_list(open(get_corpus_path('env_pos.txt')).readlines())
    env_neu = cut_sent_list(open(get_corpus_path('env_neu.txt')).readlines())
    env_neg = cut_sent_list(open(get_corpus_path('env_neg.txt')).readlines())
    price_pos = cut_sent_list(open(get_corpus_path('price_pos.txt')).readlines())
    price_neu = cut_sent_list(open(get_corpus_path('price_neu.txt')).readlines())
    price_neg = cut_sent_list(open(get_corpus_path('price_neg.txt')).readlines())
    ser_pos = cut_sent_list(open(get_corpus_path('ser_pos.txt')).readlines())
    ser_neu = cut_sent_list(open(get_corpus_path('ser_neu.txt')).readlines())
    ser_neg = cut_sent_list(open(get_corpus_path('ser_neg.txt')).readlines())

    append_dict(corpus, food_pos, 1, '味道')
    append_dict(corpus, food_neu, 0, '味道')
    append_dict(corpus, food_neg, -1, '味道')
    append_dict(corpus, env_pos, 1, '环境')
    append_dict(corpus, env_neu, 0, '环境')
    append_dict(corpus, env_neg, -1, '环境')
    append_dict(corpus, price_pos, 1, '价格')
    append_dict(corpus, price_neu, 0, '价格')
    append_dict(corpus, price_neg, -1, '价格')
    append_dict(corpus, ser_pos, 1, '服务')
    append_dict(corpus, ser_neu, 0, '服务')
    append_dict(corpus, ser_neg, -1, '服务')

    random.shuffle(corpus)

    test_point = int(len(corpus) * 0.7)
    dev_point = int(len(corpus) * 0.9)
    f_train = open('train.cor', 'w')
    f_test = open('test.cor', 'w')
    f_dev = open('dev.cor', 'w')
    for dict0 in corpus[: test_point]:
        f_train.write(dict0['text'])
        f_train.write(dict0['cate'])
        f_train.write(dict0['pola'])
    for dict0 in corpus[test_point: dev_point]:
        f_test.write(dict0['text'])
        f_test.write(dict0['cate'])
        f_test.write(dict0['pola'])
    for dict0 in corpus[dev_point:]:
        f_dev.write(dict0['text'])
        f_dev.write(dict0['cate'])
        f_dev.write(dict0['pola'])