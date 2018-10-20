from abc import ABC, abstractmethod
from typing import Any, Callable, List, Dict

import numpy as np
from keras import Model, Input
from keras.engine import Layer
from keras.layers import Embedding

from dataset import DataSet


class Feature(ABC):

    @abstractmethod
    def model(self, input: Any) -> Model: raise NotImplementedError

    @abstractmethod
    def input(self): raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: DataSet) -> np.ndarray: raise NotImplementedError

    @abstractmethod
    def name(self) -> str: raise NotImplementedError

    def _transform_by_func(self, dataset: DataSet, func: Callable[[str], Any], input3d: bool=False, input3d_dim: int=1):
        shape = [len(dataset), dataset.sentence_length()]
        if input3d: shape.append(input3d_dim)
        res: np.ndarray = np.zeros(shape, dtype='int32')
        for sent_idx, sent in enumerate(dataset.data):
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                value: str = word[self.name()]
                feature_val = func(value)
                if input3d: res[sent_idx, word_idx, :] = feature_val
                else: res[sent_idx, word_idx] = feature_val
        return res

    def _transform_by_func_to_flat_array(self, dataset: DataSet, func: Callable[[str], Any]):
        res = []
        for sent_idx, sent in enumerate(dataset.data):
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                value: str = word[self.name()]
                feature_idx = func(value)
                res.append([feature_idx])
        return res


class OneHotFeature(Feature):

    def __init__(self, name: str, alphabet: List[str], trainable: bool=True, random_weights: bool=True, input3d: bool = False):
        self.__name = name
        self.__alphabet = {val: idx for idx, val in enumerate(alphabet)}
        self.__size: int = len(self.__alphabet)
        self.__weights: np.ndarray = self.__init_weights(random_weights)
        self.__random = random_weights
        self.__trainable = trainable
        self.__input3d: bool = input3d

    def __init_weights(self, random_weights: bool=False) -> np.ndarray:
        if random_weights: return np.random.uniform(-0.5, 0.5, size=(self.__size, self.__size))
        else: return np.identity(self.__size, dtype='float32')

    def model(self, input: Any) -> Layer:
        w = [self.__weights]
        size = self.__size
        name = self.__name + '_embedding'
        return Embedding(output_dim=size, input_dim=size, weights=w, trainable=self.__trainable, name=name)(input)

    def input(self):
        return Input(shape=(None,),dtype='int32',name=self.__name + '_onehot_input')

    def transform(self, dataset: DataSet):
        return self._transform_by_func(dataset, lambda val: self.__alphabet.get(val, 1), input3d=self.__input3d)

    def transform_to_flat_array(self, dataset: DataSet, func: Callable[[str], Any] = None):
        return self._transform_by_func_to_flat_array(dataset, func if func else lambda val: self.__alphabet.get(val, 1))

    def inverse_transform(self, array: np.ndarray, dataset: DataSet, pad_symbol: str) -> List[List[str]]:
        res: List[List[str]] = []
        idx2symbol: Dict[int, str] = {idx: val for val, idx in self.__alphabet.items()}
        for sent_idx in range(array.shape[0]):
            original_size = len(dataset.data[sent_idx])
            sent: List[str] = []
            for word_idx in range(array.shape[1]):
                if word_idx >= original_size: break
                symbol_idx = np.argmax(array[sent_idx, word_idx])
                sent.append(idx2symbol[symbol_idx])
            if len(sent) < original_size:
                extend = original_size - len(sent)
                sent.extend([pad_symbol]*extend)
            res.append(sent)
        return res

    def name(self) -> str:
        return self.__name