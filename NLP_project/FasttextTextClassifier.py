# -*- coding: UTF-8 -*-
import fastText
import jieba


class FasttextTextClassifier():
    def __init__(self, train_input, test_input, output):
        '''

        :param train_input: 训练数据文件路径
        :param test_input:  验证数据文件路径
        :param output:      模型
        '''
        self.train_input = train_input
        self.test_input = test_input
        self.output = output

    def fit(self):
        '''
        模型训练
        :return:
        '''
        fastText.train_supervised(self.train_input, label="__label__").save_model(self.output)

    def load_model(self,model_path):
        '''
        文件中加载模型
        :param model_path:
        :return:
        '''
        return fastText.load_model(model_path)

    def evaluate(self):
        '''
        通过precision，recall 评估
        :return:
        '''
        fasttext_evaluate = fastText.load_model(self.output).test(self.test_input)
        rows = fasttext_evaluate[0]
        precision = fasttext_evaluate[1]
        recall = fasttext_evaluate[2]
        return precision, recall, rows

    def predict(self, text, model):
        '''
        分词后的文本内容，空格分割
        :param text:  word1 word2 word3
        :return:
        '''
        labels, probs = model.predict([text])
        label = str(labels[0][0]).replace("__label__", "")
        confidence = '%.2f' % probs[0][0]
        return label, confidence


if __name__ == "__main__":

    # 模型文件路径
    model_path = "fasttext_20190226.model"
    # 训练 && 评估
    # text_classifier = FasttextTextClassifier("data_sample/fasttext_train", "data_sample/fasttext_train",model_path)
    # text_classifier.fit()
    # precision, recall, _ = text_classifier.evaluate()
    # print("precision={0},recall={1}".format("%.2f" % precision, "%.2f" % recall))

    # 预测
    text_classifier = FasttextTextClassifier("", "", model_path)
    model = text_classifier.load_model(model_path)
    label_to_cate = {1: 'technology', 2: 'car', 3: 'entertainment', 4: 'military', 5: 'sports'}
    data = "空空作战主要提升了雷达还有电抗的性能，最突出的特点就是增加了对地精确打击能力。"
    text = " ".join(jieba.lcut(data))
    print(text)
    label, confidence = text_classifier.predict(text, model)
    print("input data={0}".format(data))
    print("category={0}".format(label_to_cate[int(label)]))
    print("confidence={0}".format(confidence))