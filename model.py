import collections
import os
import pickle
import unittest
from typing import Iterable, Dict, List, Tuple, Union, Callable

import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.engine import Layer
from keras.layers import concatenate, Bidirectional, LSTM, TimeDistributed, Dense, CuDNNLSTM, Dropout, CuDNNGRU, GRU
from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Nadam
from keras_contrib.layers import CRF

from dataset import DataSet, DataSetFeature
from feature.base import OneHotFeature, Feature
from feature.chars import CharsFeature, CharCNNFeature
from feature.embeddings import EmbeddingFeature
from metrics.conlleval import ConllevalOptions, EvalCounts, evaluate, get_metrics, report, ConllevalMetrics


class TaggingPrediction(object):

    def __init__(self, words: List[List[str]], labels_true: List[List[str]], labels_pred: List[List[str]], otag: str='O'):
        self.words = words
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.otag = otag

    def evaluate(self, ignore_tagging_scheme=False, verbose=True) -> Tuple[ConllevalMetrics, Dict[str, ConllevalMetrics]]:
        options = ConllevalOptions(boundary='EOS', delimiter='<SPACE>', otag=self.otag)
        res: EvalCounts = evaluate(self.__lines(split_sentences=False, mark_errors=False), options, ignore_tagging_scheme)
        if verbose: report(res)
        return get_metrics(res)

    def print(self, output_path: str, include_header: bool=True):
        with open(output_path, 'w', encoding='utf-8') as output_file:
            if include_header: output_file.write("value target predicted\n")
            lines: List[str] = self.__lines(split_sentences=True, mark_errors=True)
            for line in lines:
                output_file.write(line)
                output_file.write('\n')

    def __lines(self, split_sentences: bool=False, mark_errors: bool=False) -> List[str]:
        res: List[str] = []
        for sent_idx in range(len(self.words)):
            sent: List[str] = self.words[sent_idx]
            labels_true: List[str] = self.labels_true[sent_idx]
            labels_pred: List[str] = self.labels_pred[sent_idx]
            for idx in range(len(sent)):
                mark: str = '>>' if mark_errors and labels_pred[idx] != labels_true[idx] else ''
                res.append(f'{mark}{sent[idx]} {labels_true[idx]} {labels_pred[idx]}')
            if split_sentences:
                res.append('')
        return res


class TaggingModel(object):

    def __init__(self, features: Iterable[Feature], target_col: DataSetFeature, use_crf: bool=True,
                 lstm_cudnn: bool=True, lstm_size: Union[int, List]=100, lstm_layers: int=3,
                 lstm_dropout=(0.25, 0.25, 0.25), recurrent_dropout: float=0.0, use_gru: bool=False,
                 opt_clipnorm: bool=False, otag: str='O', verbose: int = 1):
        self.__labels: List[str] = target_col.alphabet
        self.__target_name: str = target_col.name
        self.__features: List[Feature] = list(features)
        self.__lstm_cudnn = lstm_cudnn
        self.__use_crf = use_crf
        self.__lstm_size = lstm_size
        self.__lstm_layers = lstm_layers
        self.__lstm_dropout: List[float] = self.__get_lstm_dropout(lstm_layers, lstm_dropout)
        self.__recurrent_dropout = recurrent_dropout
        self.__use_gru: float = use_gru
        self.__opt_clipnorm = opt_clipnorm
        self.__otag = otag
        inputs = [feature.input() for feature in features]
        feature_models = [feature.model(inputs[idx]) for idx, feature in enumerate(features)]
        self.__model: Model = self.__build(inputs, feature_models, verbose=verbose)
        lstm_dropout_size: int = len(self.__lstm_dropout)
        assert lstm_dropout_size >= (self.__lstm_layers), f"len(lstm_dropout) should be >= {lstm_dropout_size}"
        assert not (lstm_cudnn and recurrent_dropout > 0.0), "recurrent_dropout is not supported in lstm_cudnn mode"

    def __get_lstm_dropout(self, lstm_layers: int, lstm_dropout) -> List[float]:
        if isinstance(lstm_dropout, tuple): return list(lstm_dropout)
        elif isinstance(lstm_dropout, list): return lstm_dropout
        else: return [lstm_dropout] * lstm_layers

    def __build(self, inputs: List[Input], feature_models: List[Model], verbose: int = 1) -> Model:
        hidden = concatenate(feature_models, name='concat_wordrep')
        layer_func: Callable = self.__cudnn_layer if self.__lstm_cudnn else self.__noncudnn_layer
        for idx in range(self.__lstm_layers):
            dropout = 0.0
            if idx < self.__lstm_layers and idx < len(self.__lstm_dropout) and self.__lstm_dropout[idx] > 0.0:
                dropout = self.__lstm_dropout[idx]
            hidden = layer_func(hidden, idx, dropout, self.__recurrent_dropout)
        model = self.__output_crf(inputs, hidden) if self.__use_crf else self.__output_softmax(inputs, hidden)
        if verbose > 0: model.summary()
        return model

    def __cudnn_layer(self, input: any, idx: int, dropout: float, recurrent_dropout: float):
        layer = input
        if dropout > 0.0: layer = Dropout(dropout)(layer)
        cell = CuDNNGRU if self.__use_gru else CuDNNLSTM
        size: int = self.__lstm_size[idx] if isinstance(self.__lstm_size, collections.Iterable) else self.__lstm_size
        return Bidirectional(cell(size, return_sequences=True), name=f'bilstm_wordrep_nr{idx+1}')(layer)

    def __noncudnn_layer(self, input: any, idx: int, dropout: float, recurrent_dropout: float):
        cell = GRU if self.__use_gru else LSTM
        size: int = self.__lstm_size[idx] if isinstance(self.__lstm_size, collections.Iterable) else self.__lstm_size
        layer = cell(size, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
        return Bidirectional(layer, name=f'bilstm_wordrep_nr{idx+1}')(input)

    def __output_softmax(self, inputs: List[Input], input: Layer) -> Model:
        output = TimeDistributed(Dense(len(self.__labels), activation='softmax'))(input)
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss=sparse_categorical_crossentropy, optimizer=self.__optimizer())
        return model

    def __output_crf(self, inputs: List[Input], input: Layer) -> Model:
        crf = CRF(len(self.__labels), learn_mode='join', test_mode='viterbi', sparse_target=True, name='crf')
        output = crf(input)
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss=crf.loss_function, optimizer=self.__optimizer())
        return model

    def __optimizer(self):
        return Nadam() if not self.__opt_clipnorm else Nadam(clipnorm=1.)

    def __run_opts(self):
        return tf.RunOptions(report_tensor_allocations_upon_oom=True)

    def train(self, train: DataSet, valid: DataSet = None, epochs: int=50, batch_size: int=32, verbose: int=1):
        x, y = self.__transform_dataset(train)
        validation_data = self.__transform_dataset(valid) if valid is not None else None
        self.__model.fit(x=x, y=y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def __transform_dataset(self, dataset: DataSet) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        target_feature = OneHotFeature(self.__target_name, dataset.labels(self.__target_name), input3d=True)
        x: List[np.ndarray] = [feature.transform(dataset) for feature in self.__features]
        y: List[np.ndarray] = target_feature.transform(dataset)
        return x, y

    def test(self, test: DataSet, verbose: int = 1) -> TaggingPrediction:
        words: List[List[str]] = test.values('value')
        labels_true: List[List[str]] = test.values(self.__target_name)
        labels_pred: List[List[str]] = self.predict(test, string_labels=True, verbose=verbose)
        return TaggingPrediction(words, labels_true, labels_pred, otag=self.__otag)

    def predict(self, test: DataSet, string_labels: bool=False, verbose: int = 1) -> Union[np.ndarray, List[List[str]]]:
        x: List[np.ndarray] = [feature.transform(test) for feature in self.__features]
        y_pred: np.ndarray = self.__model.predict(x, verbose=verbose)
        if not string_labels: return y_pred
        target_feature = OneHotFeature(self.__target_name, test.labels(self.__target_name), input3d=True)
        return target_feature.inverse_transform(y_pred, test, self.__otag)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_NERModel__model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def save(self, output_path: str):
        if not os.path.exists(output_path): os.mkdir(output_path)
        TaggingModel.__check_isdir(output_path)
        with open(os.path.join(output_path, "model_meta.bin"), 'wb') as model_meta: pickle.dump(self, model_meta)
        self.__model.save(os.path.join(output_path, "model_weights.bin"), overwrite=True, include_optimizer=False)

    @staticmethod
    def load(input_path: str):
        TaggingModel.__check_isdir(input_path)
        with open(os.path.join(input_path, "model_meta.bin"), "rb") as model_meta:
            res: TaggingModel = pickle.load(model_meta)
            res.__model = load_model(os.path.join(input_path, "model_weights.bin"), custom_objects={"CRF": CRF})
            return res

    @staticmethod
    def __check_isdir(dir_path):
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            raise IOError(f"{dir_path} does not exist or is not a directory!")


class TaggingModelTest(unittest.TestCase):

    def test_toy_model(self):
        feature1 = self.__embedding_feature(['<pad>', '<unk>', 'ala', 'ma', 'kota'])
        feature2 = CharCNNFeature('chars', CharsFeature.default_alphabet())
        feature3 = OneHotFeature('flag', ['<pad>', '0', '1', '2'])
        features = [feature1, feature2, feature3]
        target = DataSetFeature('target', ['PER', 'LOC', 'ORG', 'O'])
        TaggingModel(features, target)

    def __embedding_feature(self, vocab: List[str], dim: int=100) -> EmbeddingFeature:
        output: str = 'temp_embedding.txt'
        with open(output, 'w', encoding='utf-8') as output_file:
            output_file.write(f'{len(vocab)} {dim}\n')
            for word in vocab:
                vector = np.zeros(shape=(dim,)) if word == '<pad>' else np.random.uniform(-0.25, 0.25, size=dim)
                arr = np.array2string(vector, formatter={'float_kind': lambda val: "%.6f" % val})
                output_file.write(word + ' ' + arr.strip('[]').replace('\n', '') + '\n')
        return EmbeddingFeature('words', output, text_format=True)
