import collections
import logging
import os
import pickle
import unittest
from io import StringIO
from typing import Iterable, Dict, List, Tuple, Union, Callable

import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from sklearn import metrics, preprocessing
from keras import Model, Input
from keras.engine import Layer
from keras.layers import concatenate, Bidirectional, LSTM, TimeDistributed, Dense, CuDNNLSTM, Dropout, CuDNNGRU, GRU, \
    SpatialDropout1D
from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Nadam
from keras_contrib.layers import CRF

from bilm.elmo_keras import WeightElmo
from callback.restore_weights import RestoreWeights
from callback.sgdr import SGDRScheduler
from dataset import DataSet, DataSetFeature
from feature.base import OneHotFeature, Feature, DocOneHotFeature
from feature.chars import CharsFeature, CharCNNFeature
from feature.embeddings import EmbeddingFeature
from layers.attention import Attention
from metrics.conlleval import ConllevalOptions, EvalCounts, evaluate, get_metrics, report, ConllevalMetrics
from utils.files import ProjectPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelParams(object):

    def __init__(self, use_crf: bool=True, lstm_cudnn: bool=True, lstm_size: Union[int, List]=100, lstm_layers: int=3,
                 lstm_dropout=(0.25, 0.25, 0.25), recurrent_dropout: float=0.0, use_gru: bool=False,
                 opt_clipnorm: bool=False, learning_rate: float=0.002, otag: str='O', verbose: int = 1):
        self.lstm_cudnn = lstm_cudnn
        self.use_crf = use_crf
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout: List[float] = self.__get_lstm_dropout(lstm_layers, lstm_dropout)
        self.recurrent_dropout = recurrent_dropout
        self.use_gru: float = use_gru
        self.opt_clipnorm = opt_clipnorm
        self.otag = otag
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.optimizer = Nadam
        self._scheduler = None
        self._scheduler_needs_validation = False
        self._early_stopping = None
        self.restore_best_weights()

    def __get_lstm_dropout(self, lstm_layers: int, lstm_dropout) -> List[float]:
        if isinstance(lstm_dropout, tuple): return list(lstm_dropout)
        elif isinstance(lstm_dropout, list): return lstm_dropout
        else: return [lstm_dropout] * lstm_layers

    def early_stopping(self, patience: int=5, restore_best_weights: bool=True):
        self._early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=patience,
                                             verbose=1, restore_best_weights=restore_best_weights)

    def restore_best_weights(self):
        self._restore_best = RestoreWeights(verbose=1)

    def reduce_lr_on_plateau_scheduler(self, factor=0.5, patience=3, min_lr=0.0002):
        self._scheduler = ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=patience,
                                            verbose=1, min_lr=min_lr)
        self._scheduler_needs_validation = True

    def sgd_with_restarts_scheduler(self, train: DataSet, batch_size: int, max_lr=0.05, min_lr=0.0002,
                                    cycle_length=3, mult_factor=2):
        steps_per_epoch = np.ceil(len(train) / batch_size)
        self._scheduler = SGDRScheduler(max_lr=max_lr, min_lr=min_lr, steps_per_epoch=steps_per_epoch,
                                        cycle_length=cycle_length, mult_factor=mult_factor)
        self._scheduler_needs_validation = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_callbacks(self, validated: bool) -> List[Callback]:
        if not validated:
            logger.warning("No validation dataset is provided, some callbacks or schedulers may not be available")
        callbacks = []
        if self._scheduler:
            if self._scheduler_needs_validation:
                if validated: callbacks.append(self._scheduler)
            else: callbacks.append(self._scheduler)
        if self._early_stopping and validated: callbacks.append(self._early_stopping)
        if self._restore_best and validated: callbacks.append(self._restore_best)
        return callbacks

    def clear_callbacks(self):
        self._scheduler = None
        self._early_stopping = None
        self._restore_best = None


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


class ClassificationPrediction(object):

    def __init__(self, labels_true: List[str], labels_pred: List[str]):
        self.labels_true = labels_true
        self.labels_pred = labels_pred

    def evaluate(self, verbose=True) -> Dict[str, float]:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(self.labels_true + self.labels_pred)
        y_true = encoder.transform(self.labels_true)
        y_pred = encoder.transform(self.labels_pred)
        acc = metrics.accuracy_score(y_true, y_pred) * 100.0
        f1 = metrics.f1_score(y_true, y_pred, average="weighted") * 100.0
        if verbose:
            print(f"classification - accuracy: {acc:.2f}%, f1: {f1:.2f}")
            print(self.__confusion_matrix(y_true, y_pred, encoder))
        return {"accuracy": acc, "f1": f1}

    def __confusion_matrix(self, y_true, y_pred, encoder: preprocessing.LabelEncoder):
        cm: List[List[float]] = metrics.confusion_matrix(y_true, y_pred)
        label_length = max([len(clazz) for clazz in encoder.classes_]) + 5
        output: StringIO = StringIO()
        output.write(" "*label_length)
        for idx in range(len(cm)): output.write("{:^8}".format(f"[{idx}]"))
        output.write("\n")
        for idx, row in enumerate(cm):
            output.write(f"{encoder.classes_[idx]} [{idx}]".rjust(label_length))
            for idx, col in enumerate(row):
                output.write("{:^8}".format(col))
            output.write("\n")
        res = output.getvalue()
        output.close()
        return res


class TaggingModel(object):

    def __init__(self, features: Iterable[Feature], target: DataSetFeature,
                 doc_target: DataSetFeature=None, params: ModelParams=None):
        if params is None: params = ModelParams()
        self.labels: List[str] = target.alphabet
        self.target_name: str = target.name
        self.doc_labels: List[str] = doc_target.alphabet if doc_target else None
        self.doc_target_name: str = doc_target.name if doc_target else None
        self.features: List[Feature] = list(features)
        self.params = params
        inputs = [feature.input() for feature in features]
        feature_models = [feature.model(inputs[idx]) for idx, feature in enumerate(features)]
        self.model: Model = self.__build(inputs, feature_models, verbose=params.verbose)
        lstm_dropout_size: int = len(self.params.lstm_dropout)
        assert lstm_dropout_size >= (self.params.lstm_layers), f"len(lstm_dropout) should be >= {lstm_dropout_size}"
        assert not (params.lstm_cudnn and params.recurrent_dropout > 0.0), "recurrent_dropout is not supported in lstm_cudnn mode"

    def __build(self, inputs: List[Input], feature_models: List[Model], verbose: int = 1) -> Model:
        hidden = concatenate(feature_models, name='concat_wordrep') if len(feature_models) > 1 else feature_models[0]
        layer_func: Callable = self.__cudnn_layer if self.params.lstm_cudnn else self.__noncudnn_layer
        for idx in range(self.params.lstm_layers):
            dropout = 0.0
            if idx < self.params.lstm_layers and idx < len(self.params.lstm_dropout) and self.params.lstm_dropout[idx] > 0.0:
                dropout = self.params.lstm_dropout[idx]
            hidden = layer_func(hidden, idx, dropout, self.params.recurrent_dropout)
        sequence_output = self.__sequence_output(hidden)
        model = self.__compile_model(inputs, hidden, sequence_output)
        if verbose > 0: model.summary()
        return model

    def __compile_model(self, inputs: List[Input], lstm: Layer, sequence_output: Layer) -> Model:
        outputs = [sequence_output]
        loss = sparse_categorical_crossentropy if not self.params.use_crf else sequence_output.loss_function
        if self.doc_target_name is not None:
            doc_output = Attention(bias=False)(lstm)
            doc_output = Dropout(0.1)(doc_output)
            doc_output = Dense(len(self.doc_labels), activation='softmax', name='doc_output')(doc_output)
            outputs.append(doc_output)
            loss = {'sequence_output': sparse_categorical_crossentropy, 'doc_output': sparse_categorical_crossentropy}
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=loss, optimizer=self.__optimizer())
        return model

    def __cudnn_layer(self, input: any, idx: int, dropout: float, recurrent_dropout: float):
        layer = input
        if dropout > 0.0: layer = SpatialDropout1D(dropout)(layer)
        cell = CuDNNGRU if self.params.use_gru else CuDNNLSTM
        size_list: bool = isinstance(self.params.lstm_size, collections.Iterable)
        size: int = self.params.lstm_size[idx] if size_list else self.params.lstm_size
        return Bidirectional(cell(size, return_sequences=True), name=f'bilstm_wordrep_nr{idx+1}')(layer)

    def __noncudnn_layer(self, input: any, idx: int, dropout: float, recurrent_dropout: float):
        cell = GRU if self.params.use_gru else LSTM
        size_list: bool = isinstance(self.params.lstm_size, collections.Iterable)
        size: int = self.params.lstm_size[idx] if size_list else self.params.lstm_size
        layer = cell(size, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
        return Bidirectional(layer, name=f'bilstm_wordrep_nr{idx+1}')(input)

    def __sequence_output(self, input: Layer) -> Layer:
        layer_name = 'sequence_output'
        if self.params.use_crf:
            learn_mode = "marginal" #'join' if self.doc_target_name is None else 'marginal'
            test_mode = "marginal" #'viterbi' if self.doc_target_name is None else 'marginal'
            crf = CRF(len(self.labels), learn_mode=learn_mode, test_mode=test_mode, sparse_target=True, name=layer_name)
            res = crf(input)
            res.loss_function = crf.loss_function
            return res
        else:
            return TimeDistributed(Dense(len(self.labels), activation='softmax'), name=layer_name)(input)

    def __optimizer(self):
        lr = self.params.learning_rate
        opt = self.params.optimizer
        return opt(lr=lr) if not self.params.opt_clipnorm else opt(clipnorm=1., lr=lr)

    def __run_opts(self):
        return tf.RunOptions(report_tensor_allocations_upon_oom=True)

    def train(self, train: DataSet, valid: DataSet = None, epochs: int=50, batch_size: int=32, verbose: int=1):
        x, y = self.__transform_dataset(train)
        validation_data = self.__transform_dataset(valid) if valid is not None else None
        callbacks = self.params.get_callbacks(valid is not None)
        self.params.clear_callbacks()
        self.model.fit(x=x, y=y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def __transform_dataset(self, dataset: DataSet) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        target_feature = OneHotFeature(self.target_name, dataset.labels(self.target_name), input3d=True)
        x: List[np.ndarray] = [feature.transform(dataset) for feature in self.features]
        y: List[np.ndarray] = target_feature.transform(dataset)
        if self.doc_target_name is not None:
            doc_target_feature = DocOneHotFeature(self.doc_target_name, dataset.labels(self.doc_target_name))
            y = [y, doc_target_feature.transform(dataset)]
        return x, y

    def test(self, test: DataSet, verbose: int = 1) -> Union[TaggingPrediction, Tuple]:
        preds: Union[tuple, List[List[str]]] = self.predict(test, string_labels=True, verbose=verbose)
        if not isinstance(preds, tuple): return self.__tagging_predictions(test, preds)
        else: return self.__tagging_predictions(test, preds[0]), self.__classification_predictions(test, preds[1])

    def __tagging_predictions(self, test: DataSet, labels_pred: List[List[str]]) -> TaggingPrediction:
        words: List[List[str]] = test.values('value')
        labels_true: List[List[str]] = test.values(self.target_name)
        return TaggingPrediction(words, labels_true, labels_pred, otag=self.params.otag)

    def __classification_predictions(self, test: DataSet, labels_pred: List[str]) -> ClassificationPrediction:
        labels_true: List[str] = test.docvalues(self.doc_target_name)
        return ClassificationPrediction(labels_true, labels_pred)

    def predict(self, test: DataSet, string_labels: bool=False, verbose: int = 1):
        x: List[np.ndarray] = [feature.transform(test) for feature in self.features]
        y_pred: Union[np.ndarray, List[np.ndarray]] = self.model.predict(x, verbose=verbose)
        if not string_labels: return y_pred
        if isinstance(y_pred, list): return self.__word_predictions(y_pred, test), self.__doc_predictions(y_pred, test)
        else: return self.__word_predictions(y_pred, test)

    def __word_predictions(self, y_pred: Union[np.ndarray, List[np.ndarray]], test: DataSet):
        predictions = y_pred[0] if isinstance(y_pred, list) else y_pred
        target_feature = OneHotFeature(self.target_name, test.labels(self.target_name), input3d=True)
        return target_feature.inverse_transform(predictions, test, self.params.otag)

    def __doc_predictions(self, y_pred: Union[np.ndarray, List[np.ndarray]], test: DataSet):
        predictions = y_pred[1] if isinstance(y_pred, list) else y_pred
        target_feature = DocOneHotFeature(self.doc_target_name, test.labels(self.doc_target_name))
        return target_feature.inverse_transform(predictions)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def save(self, output_path: str):
        if not os.path.exists(output_path): os.mkdir(output_path)
        TaggingModel.__check_isdir(output_path)
        with open(os.path.join(output_path, "model_meta.bin"), 'wb') as model_meta: pickle.dump(self, model_meta)
        self.model.save(os.path.join(output_path, "model_weights.bin"), overwrite=True, include_optimizer=False)

    @staticmethod
    def load(input_path: str):
        TaggingModel.__check_isdir(input_path)
        with open(os.path.join(input_path, "model_meta.bin"), "rb") as model_meta:
            custom_objects = {"CRF": CRF, "Attention": Attention, "WeightElmo": WeightElmo}
            res: TaggingModel = pickle.load(model_meta)
            res.model = load_model(os.path.join(input_path, "model_weights.bin"), custom_objects=custom_objects)
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
        os.environ["TEST_PATH"] = "."
        return EmbeddingFeature('words', ProjectPath("TEST_PATH", output), text_format=True)
