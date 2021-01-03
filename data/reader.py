# -*- coding: UTF-8 -*-

import os
import re
import string
import sys

from .Chatbot.RuleMatcher.answerer import Answerer
sys.path.append('..')
from utils import *

class Reader():
    def __init__(self):
        self.answerer = Answerer()
        self.mList_ = list()
        self.rList_ = list()

    def MessageToList(self, filePath):
        mList = list()
        with open(filePath, 'r', encoding='utf-8') as m:
            for line in m:
                messages = line.split('\t')
                if len(messages) != 3:
                    continue
                message = messages[2]
                if re.match('^☎ ', message):
                    continue
                if re.match('^\[.+\]', message):
                    continue
                for punc in string.punctuation:
                    message = message.replace(punc, ' ')
                message = re.sub('\s+', ' ', message).strip()
                mList.append(message)
        if not mList:
            raise Exception('No Available Messages')
        return mList

    def ReplyToList(self, mList):
        rList = list()
        for i, message in enumerate(mList):
            reply, _ = self.answerer.getResponse(message)
            rList.append(reply[0])
            if i%1000==0:
                print('{} message has been replied.'.format(i))
        return rList
    
    def ListNorm(self, List, maxLength = 30):
        List_ = list()
        for sentence in List:
            sentence_ = Sentence()
            sentence_.addGram(Dictionary.BOS)
            for i, word in enumerate(sentence):
                if i >= maxLength:
                    continue
                sentence_.addGram(self.dict.convertTokenToGram(word))
            if sentence_.length() < maxLength:
                sentence_.addGram(Dictionary.EOS)
            List_.append(sentence_)
        return List_
    
    def getDict(self, filePath, dict = None):
        if not os.path.isfile(filePath):
            raise Exception('File \'{}\' Not Found'.format(filePath))
        mListPath = 'data/messageList.txt'
        rListPath = 'data/replyList.txt'
        if os.path.isfile(mListPath) and os.path.isfile(rListPath):
            m = open(mListPath, 'r', encoding='utf-8')
            r = open(rListPath, 'r', encoding='utf-8')
            mList = m.read().splitlines()
            rList = r.read().splitlines()
        else:
            mList = self.MessageToList(filePath)
            rList = self.ReplyToList(mList)
            m = open(mListPath, 'w')
            r = open(rListPath, 'w')
            for message in mList:
                m.write(message)
                m.write('\n')
            for reply in rList:
                r.write(reply)
                r.write('\n')
        m.close()
        r.close()
        if not dict:
            dict = Dictionary(''.join(mList+rList))
        self.dict = dict
        self.mList_ = self.ListNorm(mList)
        self.rList_ = self.ListNorm(rList)
        return dict

    def getSentence(self, index):
        if index >= len(self.mList_):
            raise Exception('Index out of range')
        return self.mList_[index]

