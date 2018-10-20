import gzip
import os
from typing import Any, Tuple, Dict, Callable

import numpy as np
from gensim.models import KeyedVectors
from keras import Input
from keras.engine import Layer

from dataset import DataSet
from feature.base import Feature


class EmbeddingFeature(Feature):

    def __init__(self, name: str, embedding_path: str, text_format: bool=False, to_lowercase: bool=True,
                 return_vectors: bool=False, trainable: bool=False):
        self.__embedding_path = embedding_path
        self.__text_format = text_format
        self.__embedding = self.__load_embedding(embedding_path, text_format)
        self.__name = name
        self.__to_lower: bool = to_lowercase
        self.__return_vectors: bool = return_vectors
        self.__trainable: bool = trainable
        assert not (return_vectors and trainable), "feature cannot be trainable when return_vectors == True"

    def model(self, input: Any) -> Layer:
        if self.__return_vectors: return input
        else: return self.__embedding.wv.get_keras_embedding(train_embeddings=self.__trainable)(input)

    def input(self):
        dtype: str = 'float32' if self.__return_vectors else 'int32'
        shape: Tuple = (None, self.__embedding.vector_size) if self.__return_vectors else (None,)
        return Input(shape=shape,dtype=dtype,name=self.__name + '_embedding_input')

    def __load_embedding(self, path: str, text_format: bool=False) -> KeyedVectors:
        return KeyedVectors.load(path) if not text_format else KeyedVectors.load_word2vec_format(path, binary=False)

    def __vocab_index(self, word: str) -> int:
        val: str = word.lower() if self.__to_lower else word
        return self.__embedding.wv.vocab.get(val, self.__embedding.wv.vocab['<unk>']).index

    def __vocab_vector(self, word: str) -> np.ndarray:
        return self.__embedding.wv.syn0[self.__vocab_index(word)]

    def transform(self, dataset: DataSet):
        transform = self.__vocab_vector if self.__return_vectors else self.__vocab_index
        dim: int = self.__embedding.vector_size
        return self._transform_by_func(dataset, transform, input3d=self.__return_vectors, input3d_dim=dim)

    def name(self) -> str:
        return self.__name

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_EmbeddingFeature__embedding"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not os.path.exists(self.__embedding_path):
            embedding_path = os.environ.get("EMBEDDING_PATH", None)
            if not embedding_path:
                error = "deserialized path {} does not exist, consider overriding it by EMBEDDING_PATH env variable"
                raise IOError(error.format(self.__embedding_path))
            else: self.__embedding_path = embedding_path
        self.__embedding = self.__load_embedding(self.__embedding_path, self.__text_format)


class CompressedEmbeddingFeature(Feature):

    def __init__(self, name: str, vocab_path: str, embedding_path: str, to_lowercase: bool=True):
        self.__name: str = name
        self.vocab_path: str = vocab_path
        self.embedding_path: str = embedding_path
        self.to_lower: bool = to_lowercase
        self.vocab: Dict[str, int] = self.__load_vocab(vocab_path)
        embedding = np.load(embedding_path)
        self.codes: np.ndarray = embedding[embedding.files[0]]
        self.codebook: np.ndarray = embedding[embedding.files[1]]
        self.m = self.codes.shape[1]
        self.k = int(self.codebook.shape[0] / self.m)
        self.dim: int = self.codebook.shape[1]

    def model(self, input: Any) -> Layer:
        return input

    def input(self):
        return Input(shape=(None, self.dim),dtype=np.float32,name=self.__name + '_compressed_embedding_input')

    def __load_vocab(self, vocab_path: str) -> Dict[str, int]:
        open_func: Callable = gzip.open if vocab_path.endswith(".gz") else open
        with open_func(vocab_path, "rt", encoding="utf-8") as input_file:
            return {line.strip():idx for idx, line in enumerate(input_file)}

    def vocab_vector(self, word: str):
        if word == "<pad>": return np.zeros(self.dim)
        val: str = word.lower() if self.to_lower else word
        index: int = self.vocab.get(val, self.vocab["<unk>"])
        codes = self.codes[index]
        code_indices = np.array([idx * self.k + offset for idx, offset in enumerate(np.nditer(codes))])
        return np.sum(self.codebook[code_indices], axis=0)

    def transform(self, dataset: DataSet):
        return self._transform_by_func(dataset, self.vocab_vector, input3d=True, input3d_dim=self.dim)

    def name(self) -> str:
        return self.__name