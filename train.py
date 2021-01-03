# -*- coding: UTF-8 -*-
import random
from data.reader import Reader
from model.model import SeqToSeqModel

epoch = 30

reader = Reader()
dict = reader.getDict('data/history.txt')
Model = SeqToSeqModel(dict.size)

for i in range(epoch):
    Model.train(reader.mList_, reader.rList_)
    Model.saveWeights()

sentence = random.choice(reader.mList_)
response = Model.predict(sentence, dict)
print('[Sample Sentence] {}'.format(sentence))
print('[Sample Response] {}\n'.format(response))
