import os
from typing import Any

import numpy as np
from keras import Input
from keras.layers import Lambda

from bilm import Batcher
from bilm.elmo import all_layers
from bilm.elmo_keras import WeightElmo
from dataset import DataSet
from feature.base import Feature
from utils.files import ProjectPath


class ELMoEmbeddingFeature(Feature):

    def __init__(self, name: str, embedding_dir: ProjectPath):
        self.__name = name
        self.__embedding_dir: ProjectPath = embedding_dir
        self.__batcher = self.__create_batcher(embedding_dir)

    def input(self):
        return Input(shape=(None,50),dtype=np.int32,name=self.__name + '_elmo_embedding_input')

    def model(self, input: Any):
        options_file: str = self.__embedding_dir.join("options.json").get()
        weight_file: str = self.__embedding_dir.join("weights.hdf5").get()
        def __lambda_layer(x):
            import tensorflow as tf
            from bilm import BidirectionalLanguageModel, weight_layers
            x_input = tf.cast(x, tf.int32)
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                bilm = BidirectionalLanguageModel(options_file, weight_file)
                embedding_op = bilm(x_input)
                return all_layers(embedding_op)
        embeddings = Lambda(__lambda_layer, name=self.__name + '_elmo_lambda_layer')(input)
        return WeightElmo()(embeddings)

    def __create_batcher(self, embedding_dir: ProjectPath) -> Batcher:
        vocab_path: str = embedding_dir.join("vocabulary.txt").get()
        return Batcher(vocab_path, 50)

    def name(self) -> str:
        return self.__name

    def transform(self, dataset: DataSet):
        text = []
        for sent in dataset.data:
            sent_text = []
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                sent_text.append(word[self.__name])
            text.append(sent_text)
        return self.__batcher.batch_sentences(text, padding=dataset.sentence_length())

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_ELMoEmbeddingFeature__batcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__batcher = self.__create_batcher(self.__embedding_dir)