#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import preprocessor.builddataset as bd
import preprocessor.buildpretrainemb as bpe
import utils.statisticsdata as sd
from config import Config
from data.mingluedata import MingLueData
from utils.trainhelper import accuracy, model_selector, do_eval, build_element_vec

def main(model_id, use_element, is_save):
    config = Config()
    print("epoch num: ", config.epoch_num)
    config.use_element = use_element
    print("loading data...")
    # 原始数据 切分3列-list id  data  label
    ids, data, labels = bd.load_data(config.data_path)
    train_ids, valid_ids = bd.split_data(ids, radio=0.7)
    train_data, valid_data = bd.split_data(data, radio=0.7)
    train_labels, valid_labels = bd.split_data(labels, radio=0.7)

    # 求数据中所有词汇个数
    total_vocab_size = sd.count_vocab_size(data)
    print("total vocab size", total_vocab_size)
    print("load word2index")
    dict_word2index = bpe.load_pickle(config.word2index_path)
    # print(len(dict_word2index))

    train_ids, train_X, train_y = bd.build_dataset(train_ids, train_data, train_labels, dict_word2index,
                                                   max_text_len=config.max_text_len)
    print(train_ids[0:4])
    print(train_X[0:4])
    print(train_y[0:4])
    valid_ids, valid_X, valid_y = bd.build_dataset(valid_ids, valid_data, valid_labels, dict_word2index,
                                                   max_text_len=config.max_text_len)
    print("trainset size:", len(train_ids))
    print("validset size:", len(valid_ids))

    dataset_train = MingLueData(train_ids, train_X, train_y)
    dataset_valid = MingLueData(valid_ids, valid_X, valid_y)

    batch_size = config.batch_size
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

    valid_loader = DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

    config.vocab_size = len(dict_word2index)
    print('config vocab size:', config.vocab_size)
    model = model_selector(config, model_id, use_element)
    if config.has_cuda:
        model = model.cuda()

    loss_weight = torch.FloatTensor(config.loss_weight_value)
    loss_weight = loss_weight + 1 - loss_weight.mean()
    print("loss weight:", loss_weight)
    loss_fun = nn.CrossEntropyLoss(loss_weight.cuda())
    optimizer = model.get_optimizer(config.learning_rate,config.learning_rate2, config.weight_decay)
    print("training...")
    weight_count = 0
    max_score = 0
    total_loss_weight = torch.FloatTensor(torch.zeros(8))
    for epoch in range(config.epoch_num):
        print("lr:", config.learning_rate, "lr2:", config.learning_rate2)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            ids, texts, labels = data
            if config.has_cuda:
                inputs, labels = Variable(texts.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(texts), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % config.step == config.step - 1:
                if epoch % config.epoch_step == config.epoch_step - 1:
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu().numpy().tolist()
                    running_acc = accuracy(predicted, labels.data.cpu().numpy())
                    print('[%d, %5d] loss: %.3f, acc: %.3f' %
                          (epoch + 1, i + 1, running_loss / config.step, running_acc))
                running_loss = 0.0

        if is_save != 'y' and epoch % config.epoch_step == config.epoch_step - 1:
            print("predicting...")
            loss_weight, score = do_eval(valid_loader, model, model_id, config.has_cuda)
            if score >= 0.478 and score > max_score:
                max_score = score
                save_path = config.model_path + "." + str(score) + "." + config.model_names[model_id]
                torch.save(model.state_dict(), save_path)

            if epoch >= 3:
                weight_count += 1
                total_loss_weight += loss_weight
                print("avg_loss_weight:", total_loss_weight / weight_count)

        if epoch >= config.begin_epoch - 1:
            if epoch >= config.begin_epoch and config.learning_rate2 == 0:
                config.learning_rate2 = 2e-4
            elif config.learning_rate2 > 0:
                config.learning_rate2 *= config.lr_decay
                if config.learning_rate2 <= 1e-5:
                    config.learning_rate2 = 1e-5
            config.learning_rate = config.learning_rate * config.lr_decay
            optimizer = model.get_optimizer(config.learning_rate, config.learning_rate2,config.weight_decay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--use-element", type=str)
    parser.add_argument("--is-save", type=str)
    args = parser.parse_args()

    if args.use_element == 'y':
        use_element = True
    else:
        use_element = False
    main(args.model_id, use_element, args.is_save)
