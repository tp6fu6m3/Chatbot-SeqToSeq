# -*- coding: UTF-8 -*-
from enum import Enum

class TermFrequency():
    def __init__(self, tokenList):
        self.countingMap = dict()
        self.rankingMap = dict()
        for token in tokenList:
            if token in self.countingMap:
                self.countingMap[token] += 1
            else:
                self.countingMap[token] = 1
        rankingList = sorted(self.countingMap.items(), key = lambda item: item[1], reverse = True)
        for rank, item in enumerate(rankingList):
            token = item[0]
            self.rankingMap[token] = rank

    def getTermFrequency(self, token):
        if token in self.countingMap:
            return self.countingMap[token]
        return 0

    def isTokenRecognized(self, token):
        if token in self.countingMap:
            return True
        return False

    def getTermRank(self, token):
        if not self.isTokenRecognized(token):
            raise Exception('Token \'{}\' Not Recognized'.format(token))
        return self.rankingMap[token]

class TokenType(Enum):
    RESERVED = 0
    BOS = 1
    EOS = 2
    UNKNOWN = 3
    NONGRAMCOUNT = 4
    GRAM = 5

class Gram():
    def __init__(self, tokenType, index, token = ''):
        self.tokenType = tokenType
        self.index = index
        self.token = token

    def __repr__(self):
        if self.tokenType == TokenType.BOS:
            return '<BOS>'
        if self.tokenType == TokenType.EOS:
            return '<EOS>'
        if self.tokenType == TokenType.UNKNOWN:
            return '<UNKNOWN>'
        if self.tokenType == TokenType.RESERVED or self.tokenType == TokenType.NONGRAMCOUNT:
            raise Exception('Unexpcted Behavior')
        return self.token

class Dictionary():
    BOS = Gram(TokenType.BOS, TokenType.BOS.value)
    EOS = Gram(TokenType.EOS, TokenType.EOS.value)
    UNKNOWN = Gram(TokenType.UNKNOWN, TokenType.UNKNOWN.value)

    def __init__(self, tokenList, maxSize = 2500):
        self.mapping = dict()
        termFrequency = TermFrequency(tokenList)
        for token in tokenList:
            termRank = termFrequency.getTermRank(token)
            indexToApply = termRank + TokenType.NONGRAMCOUNT.value
            if indexToApply >= maxSize:
                continue
            self.mapping[token] = Gram(TokenType.GRAM, indexToApply, token)
            self.mapping[indexToApply] = Gram(TokenType.GRAM, indexToApply, token)
        self.size = len(self.mapping) // 2 + TokenType.NONGRAMCOUNT.value

    def convertTokenToGram(self, token):
        if token in self.mapping:
            return self.mapping[token]
        return Dictionary.UNKNOWN

    def convertIndexToGram(self, index):
        if index == Dictionary.BOS.index:
            return Dictionary.BOS
        if index == Dictionary.EOS.index:
            return Dictionary.EOS
        if index == Dictionary.UNKNOWN.index:
            return Dictionary.UNKNOWN
        if index in self.mapping:
            return self.mapping[index]
        return Dictionary.UNKNOWN

class Sentence():
    def __init__(self):
        self.gramList = list()
    def __repr__(self):
        output = ''
        for gram in self.gramList:
            output += str(gram)
        return output

    def addGram(self, gram):
        self.gramList.append(gram)
    def length(self):
        return len(self.gramList)

