from typing import Dict, List, Any

import numpy as np
from keras import Input
from keras.engine import Layer
from keras.initializers import RandomUniform
from keras.layers import TimeDistributed, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, CuDNNLSTM, \
    SpatialDropout1D

from dataset import DataSet
from feature.base import Feature


class CharsFeature(Feature):

    def input_size(self) -> int: raise NotImplementedError

    def alphabet(self) -> Dict[str, int]: raise NotImplementedError

    def input(self):
        return Input(shape=(None,self.input_size(),),name=self.name() + '_chars_input')

    def transform(self, dataset: DataSet):
        res: np.ndarray = np.zeros((len(dataset), dataset.sentence_length(), self.input_size()), dtype='int32')
        for sent_idx, sent in enumerate(dataset.data):
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                value: str = word[self.name()]
                for char_idx, char in enumerate(value):
                    if char_idx >= self.input_size(): break
                    res[sent_idx, word_idx, char_idx] = self.alphabet().get(char, self.alphabet().get("<unk>"))
        return res

    @staticmethod
    def default_alphabet():
        res = ['<pad>', '<unk>']
        res += list(" 0123456789")
        res += list("aąbcćdeęfghijklmnńoópqrsśtuvwxyzźżAĄBCĆDEĘFGHIJKLMNŃOÓPQRSŚTUVWXYZŹŻ")
        res += list(".,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|")
        return res


class CharCNNFeature(CharsFeature):

    def __init__(self, name: str, alphabet: List[str], input_size: int=52,
                 embedding_size: int=30, filters: int=30, dropout: float=0.5):
        self.__name = name
        self.__input_size = input_size
        self.__alphabet = {val: idx for idx, val in enumerate(alphabet)}
        self.__dropout = dropout
        self.__embedding_size = embedding_size
        self.__filters = filters

    def model(self, input: Any) -> Layer:
        size = len(self.alphabet())
        initializer = RandomUniform(minval=-0.5, maxval=0.5)
        embedding_step = Embedding(size, self.__embedding_size, embeddings_initializer=initializer)
        embedding = TimeDistributed(embedding_step,  name=self.__name)(input)
        embedding = Dropout(self.__dropout, name=self.__name + '_inner_dropout')(embedding)
        conv = Conv1D(kernel_size=3, filters=self.__filters, padding='same', activation='tanh', strides=1)
        conv = TimeDistributed(conv, name=self.__name + '_conv1d')(embedding)
        conv = TimeDistributed(MaxPooling1D(52), name=self.__name + '_maxpool')(conv)
        output = TimeDistributed(Flatten(), name=self.__name + '_flatten')(conv)
        output = SpatialDropout1D(self.__dropout, name=self.__name + '_output_dropout')(output)
        return output

    def name(self) -> str:
        return self.__name

    def input_size(self) -> int:
        return self.__input_size

    def alphabet(self) -> Dict[str, int]:
        return self.__alphabet


class CharBiLSTMFeature(CharsFeature):

    def __init__(self, name: str, alphabet: List[str], input_size: int=52, dropout: float=0.5):
        self.__name = name
        self.__alphabet = {val: idx for idx, val in enumerate(alphabet)}
        self.__input_size = input_size
        self.__droput = dropout

    def model(self, input: Any) -> Layer:
        size = len(self.alphabet())
        initializer = RandomUniform(minval=-0.5, maxval=0.5)
        embedding = TimeDistributed(Embedding(size, 50, embeddings_initializer=initializer))(input)
        embedding = SpatialDropout1D(self.__droput)(embedding)
        output = TimeDistributed(Bidirectional(CuDNNLSTM(100)))(embedding)
        output = SpatialDropout1D(self.__droput)(output)
        return output

    def name(self) -> str:
        return self.__name

    def input_size(self) -> int:
        return self.__input_size

    def alphabet(self) -> Dict[str, int]:
        return self.__alphabet