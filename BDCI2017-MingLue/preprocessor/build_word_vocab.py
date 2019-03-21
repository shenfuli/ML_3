# -*-coding:utf-8-*-
__author__ = 'shenfuli'
'''
    根据预处理的数据，构建word2index,index2word 
'''
import sys
sys.path.append("./")
import preprocessor.builddataset as bd
import preprocessor.buildpretrainemb as bde
from config import Config

def main():
    config = Config()
    print("loading data...")
    ids, data, labels = bd.load_data("./corpus/seg_train.txt")
    count, dict_word2index, dict_index2word = bd.build_vocabulary(data, min_count=config.min_count)
    print("save word2index and index2word")
    bde.save_dict(dict_word2index,config.word2index_path)
    bde.save_dict(dict_index2word,config.index2word_path)
    print("load word2index and index2word")
    print(bde.load_pickle(config.word2index_path))
    print(bde.load_pickle(config.index2word_path))
if __name__ == "__main__":
    main()
