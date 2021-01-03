# -*- coding: UTF-8 -*-

import random
from data.reader import Reader
from model.model import SeqToSeqModel

reader = Reader()
dict = reader.getDict('data/history.txt')
Model = SeqToSeqModel(dict.size)
Model.loadWeights('model/model.h5')

for i in range(10):
    sentence = random.choice(reader.mList_)
    response = Model.predict(sentence, dict)
    print('[Sample Sentence] {}'.format(sentence))
    print('[Sample Response] {}\n'.format(response))