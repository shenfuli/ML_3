# -*-coding:utf-8-*-
__author__ = 'shenfuli'
import sys
sys.path.append("./")
from collections import Counter
import preprocessor.buildpretrainemb as bde
from config import Config

def build_vocabulary(data, min_count=3):
    """
    基于所有数据构建词表
    """
    stopwords_data_path = "data/stopwords.txt"
    # stopwords_data_df = pd.read_csv(stopwords_data_path, encoding="utf-8", sep="\t", index_col=None, quoting=3,names=["stopword"])


    # add <PAD> for embedding
    word_count = [('<UNK>', -1), ('<PAD>', -1)]
    words = []
    for word in data:
        words.append(word)
    counter = Counter(words)
    counter_list = counter.most_common()
    for word, count in counter_list:
        if count >= min_count:
            word_count.append((word, count))
    dict_word2index = dict()
    for word, _ in word_count:
        dict_word2index[word] = len(dict_word2index)
    dict_index2word = dict(zip(dict_word2index.values(), dict_word2index.keys()))
    print("vocab size:", len(word_count))
    return word_count,dict_word2index,dict_index2word


if __name__ == "__main__":
    config = Config()
    data = ['公诉', '机关', '莆田市', '荔城区', '荔城区', '荔城区']
    word_count,dict_word2index,dict_index2word = build_vocabulary(data)
    bde.save_dict(dict_word2index,config.word2index_path)
    bde.save_dict(dict_index2word,config.index2word_path)

    print(bde.load_pickle(config.word2index_path))
    print(bde.load_pickle(config.index2word_path))