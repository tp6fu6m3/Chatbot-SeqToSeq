# -*- coding: UTF-8 -*-

import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
sys.path.append("..")

from tensorflow.keras.layers import Embedding, LSTM, Input, TimeDistributed, Dense, dot, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import Dictionary, Sentence

class SeqToSeqModel():
    def __init__(self, dictionarySize = 2500, sentenceLength = 30):
        # settings
        self.dictionarySize = dictionarySize
        self.sentenceLength = sentenceLength
        # keras overall model
        embedding = Embedding(dictionarySize, 128, mask_zero = True, input_length = None)
        encoder = LSTM(256, return_sequences = True, return_state = True)
        decoder = LSTM(256, return_sequences = True, return_state = True)
        classifierLayer1 = TimeDistributed(Dense(256, activation = 'tanh'))
        classifierLayer2 = TimeDistributed(Dense(dictionarySize, activation = 'softmax'))
        questions = Input(shape = (None,), dtype = 'int32')
        answers = Input(shape = (None,), dtype = 'int32')
        embeddedQuestions = embedding(questions)
        embeddedAnswers = embedding(answers)
        encoded, h, c = encoder(embeddedQuestions)
        decoded, _, _ = decoder(embeddedAnswers, initial_state = [h, c])
        attention = Activation('softmax')(dot([encoded, decoded], axes = [2, 2]))
        context = dot([attention, encoded], axes = [2, 1])
        features = concatenate([decoded, context])
        distributions = classifierLayer2(classifierLayer1(features))
        self.kerasOverallModel = Model([questions, answers], distributions)
        self.kerasOverallModel.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', sample_weight_mode = 'temporal')
        # keras model interfaces
        self.kerasEncoderModel = Model(questions, [encoded, h, c])
        encoded = Input(shape = (None, 256))
        hMemCells = Input(shape = (256,))
        cMemCells = Input(shape = (256,))
        decoded, h, c = decoder(embeddedAnswers, initial_state = [hMemCells, cMemCells])
        attention = Activation('softmax')(dot([encoded, decoded], axes = [2, 2]))
        context = dot([attention, encoded], axes = [2, 1])
        features = concatenate([decoded, context])
        distributions = classifierLayer2(classifierLayer1(features))
        self.kerasDecoderModel = Model([answers, encoded, hMemCells, cMemCells], [distributions, h, c])

    def train(self, questionSentences, answerSentences):
        # check size match
        if len(questionSentences) != len (answerSentences):
            raise Exception('Training Data Size Not Matched')
        # empty lists for questions and answers
        questionsToTrain = list()
        answersToTrain = list()
        distributionsLabeled = list()
        sampleWeights = list()
        for questionSentence in questionSentences:
            questionsToTrain.append(list(map(lambda gram: gram.index, questionSentence.gramList)))
        for answerSentence in answerSentences:
            answerToTrain = list(map(lambda gram: gram.index, answerSentence.gramList))
            answersToTrain.append(answerToTrain)
            distributionsLabeled.append(answerToTrain[1:])
            sampleWeights.append([math.log(len(answerToTrain))] * len(answerToTrain))
        # pad the sequences and prepare labels
        questionsToTrain = pad_sequences(questionsToTrain, maxlen = self.sentenceLength, padding = 'post')
        answersToTrain = pad_sequences(answersToTrain, maxlen = self.sentenceLength, padding = 'post')
        distributionsLabeled = pad_sequences(distributionsLabeled, maxlen = self.sentenceLength, padding = 'post')
        distributionsLabeled = to_categorical(distributionsLabeled, num_classes = self.dictionarySize)
        distributionsLabeled = distributionsLabeled.reshape(-1, self.sentenceLength, self.dictionarySize)
        sampleWeights = pad_sequences(sampleWeights, maxlen = self.sentenceLength, padding = 'post')
        # train the model using API fit()
        dataToTrain = [questionsToTrain, answersToTrain]
        self.kerasOverallModel.fit(dataToTrain, distributionsLabeled, 64, 1, verbose = 1, shuffle = True, sample_weight = sampleWeights)

    def predict(self, questionSentence, dictionaryToUse):
        # pad the sequence and allocate answer
        questionsToPred = [list(map(lambda gram: gram.index, questionSentence.gramList))]
        questionsToPred = pad_sequences(questionsToPred, maxlen = self.sentenceLength, padding = 'post')
        # init predicted answer
        encoded, h, c = self.kerasEncoderModel.predict(questionsToPred)
        possibleCandidates = [[[Dictionary.BOS.index], 0.0]]
        answersPredicted = np.zeros(questionsToPred.shape)
        answersPredicted[0, 0] = Dictionary.BOS.index
        # start collecting
        for idx in range(self.sentenceLength - 1):
            candidates = list()
            for candidate in possibleCandidates:
                sentence, score = candidate
                for posIdx, gramIdx in enumerate(sentence):
                    answersPredicted[0, posIdx] = gramIdx
                distributions, _, _ = self.kerasDecoderModel.predict([answersPredicted, encoded, h, c])
                distribution = distributions[0, idx]
                for gramIdx, gramProb in enumerate(distribution):
                    newCandidate = [sentence + [gramIdx], score - np.log(gramProb)]
                    candidates.append(newCandidate)
            possibleCandidates = sorted(candidates, key = lambda candidate: candidate[1])[:5]
        # best candidate
        sentence = possibleCandidates[0][0]
        sentenceToReturn = Sentence()
        for gramIdx in sentence:
            if gramIdx == Dictionary.EOS.index:
                break
            sentenceToReturn.addGram(dictionaryToUse.convertIndexToGram(gramIdx))
        sentenceToReturn.addGram(Dictionary.EOS)
        return sentenceToReturn

    def saveWeights(self, filePath = 'model/model.h5'):
        self.kerasOverallModel.save_weights(filePath)

    def loadWeights(self, filePath = 'model/model.h5'):
        self.kerasOverallModel.load_weights(filePath)

