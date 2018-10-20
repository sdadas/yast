import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import namedtuple
from tempfile import TemporaryDirectory
from typing import Dict, List

from dataset import DataSet
from feature.base import OneHotFeature, Feature
from feature.chars import CharBiLSTMFeature, CharsFeature, CharCNNFeature
from feature.elmo import ELMoEmbeddingFeature
from feature.embeddings import EmbeddingFeature, CompressedEmbeddingFeature
from model import TaggingModel, TaggingPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DataPaths = namedtuple("DataPaths", ["meta_path", "train_path", "test_path", "valid_path"])

class AbstractRunner(ABC):

    def __init__(self, base_path: str, config: Dict[str, any]=None, config_path: str=None):
        self.base_path = base_path
        if config is not None: self.config = config
        else: self.config = self.__create_config(config_path) if config_path else self.default_config()
        self.config_name = os.path.basename(config_path) if config_path else f"runner_{time.time()}.json"
        self.tempdir: TemporaryDirectory = TemporaryDirectory()
        self.output: Dict[str, any] = {"config": self.config, "folds": []}
        self.global_metrics: Dict[str, float] = {}
        self.class_metrics: Dict[str, Dict[str, float]] = {}

    def __create_config(self, config_path: str) -> Dict[str, any]:
        with open(config_path, 'r', encoding='utf-8') as json_file: return json.load(json_file)

    def __create_features(self, dataset: DataSet, base_path: str, paths: DataPaths):
        conf = self.get_subconfig("input")
        embedding_path = os.path.join(base_path, conf.get("embedding_path"))
        embedding_col = conf.get("embedding_col", "base")
        embedding_feature = conf.get("embedding_feature", "default")
        res = []
        if embedding_feature == "default":
            etrain: bool = conf.get("embedding_trainable", False)
            res.append(EmbeddingFeature(embedding_col, embedding_path, trainable=etrain, return_vectors=(not etrain)))
        elif embedding_feature == "compressed":
            vocab_path = embedding_path.replace("compressed.npz", "vocab.gz")
            res.append(CompressedEmbeddingFeature(embedding_col, vocab_path, embedding_path))
        elif embedding_feature == "none":
            pass
        else: raise NotImplementedError
        char_module = conf.get("char_feature", "cnn")
        if char_module == "lstm":
            res.append(CharBiLSTMFeature("value", CharsFeature.default_alphabet()))
        elif char_module == "cnn":
            res.append(CharCNNFeature("value", CharsFeature.default_alphabet()))
        elif char_module == "none":
            pass
        else: raise NotImplementedError
        elmo_enabled: bool = conf.get("elmo.enabled", False)
        if elmo_enabled:
            elmo = ELMoEmbeddingFeature("value", os.path.join(base_path, conf.get("elmo.dir_path")))
            res.append(elmo)
        feature_names = conf.get("features", [])
        for feature_name in feature_names:
            feature: Feature = self.create_feature(feature_name, dataset, base_path, paths)
            if feature is not None: res.append(feature)
        return res

    def run(self):
        paths: List[DataPaths] = self.create_cross_validation_paths(self.base_path)
        logger.info("Found %d data folds for cross validation", len(paths))
        for idx, fold in enumerate(paths):
            logger.info("Traning model on fold %s [%d/%d]", fold.train_path, (idx+1), len(paths))
            self.__run_fold(fold)
        self.after_run()

    def __run_fold(self, fold: DataPaths):
        conf = self.get_subconfig("input")
        padding: int = conf.get("padding")
        train: DataSet = DataSet(fold.train_path, fold.meta_path, padding=padding)
        test: DataSet = DataSet(fold.test_path, fold.meta_path, padding=padding)
        valid: DataSet = DataSet(fold.valid_path, fold.meta_path, padding=padding) if fold.valid_path else None
        features: List[Feature] = self.__create_features(train, self.base_path, fold)
        model = TaggingModel(features, train.column(self.config["input.target_col"]), **self.get_subconfig("model"))
        model.train(train, valid=valid, **self.get_subconfig("train"))
        pred: TaggingPrediction = model.test(test)
        self.after_fold(model, pred, fold)
        del model, features

    def after_fold(self, model: TaggingModel, pred: TaggingPrediction, fold: DataPaths):
        if self.config.get("output.predictions", False):
            fold_predictions = f"{uuid.uuid4().hex}_predictions.txt"
            pred.print(os.path.join(self.tempdir.name, fold_predictions), include_header=False)
        fold_metrics: List[any] = self.output["folds"]
        ignore_tagging_scheme: bool = self.config.get("input.ignore_tagging_scheme", False)
        global_metrics, class_metrics = pred.evaluate(ignore_tagging_scheme=ignore_tagging_scheme)
        global_metrics = global_metrics._asdict()
        class_metrics = {key: val._asdict() for key, val in class_metrics.items()}
        fold_metrics.append({"metrics": {"global": global_metrics, "class": class_metrics}})

    def after_run(self):
        output_dir = self.config.get("output.path")
        if not output_dir: return
        output_path = os.path.join(self.base_path, output_dir)
        assert not os.path.exists(output_path) or os.path.isdir(output_path), "not a directory " + output_path
        if not os.path.exists(output_path): os.mkdir(output_path)
        self.__summarize_metrics()
        if self.config.get("output.summary", True): self.__write_metrics(output_path)
        if self.config.get("output.predictions", False): self.__write_concatenated_predictions(output_path)
        self.tempdir.cleanup()

    def __write_concatenated_predictions(self, output_path: str):
        temp: str = self.tempdir.name
        file_names: List[str] = [os.path.join(temp, f) for f in os.listdir(temp) if os.path.isfile(os.path.join(temp, f))]
        output_file_name: str = self.config_name + "_predictions.txt"
        with open(os.path.join(output_path, output_file_name), 'w', encoding='utf-8') as output_file:
            output_file.write("value target predicted\n")
            for file_name in file_names:
                with open(file_name, 'r', encoding='utf-8') as input_file:
                    for line in input_file: output_file.write(line)

    def __write_metrics(self, output_path: str):
        with open(os.path.join(output_path, self.config_name), 'w', encoding='utf-8') as output_file:
            json.dump(self.output, output_file)

    def __summarize_metrics(self):
        fold_all: List[Dict[str, float]] = [fold["metrics"]["global"] for fold in self.output["folds"]]
        for fold_metrics in fold_all: self.__add_metrics(self.global_metrics, fold_metrics)
        class_all: List[Dict[Dict[str, float]]] = [fold["metrics"]["class"] for fold in self.output["folds"]]
        for fold_class_metrics in class_all:
            for class_name, fold_class_dict in fold_class_metrics.items():
                class_dict = self.class_metrics.get(class_name, dict())
                self.__add_metrics(class_dict, fold_class_dict)
                self.class_metrics[class_name] = class_dict

        fold_num: int = len(fold_all)
        self.__finalize_metrics(self.global_metrics, fold_num)
        for metrics in self.class_metrics.values(): self.__finalize_metrics(metrics, fold_num)
        self.output["metrics"] = {"global": self.global_metrics, "class": self.class_metrics}

    def __add_metrics(self, first: Dict[str, float], second: Dict[str, float]):
        for key, val in second.items():
            val_sum = first.get(key, 0.0)
            val_sum += val
            first[key] = val_sum

    def __finalize_metrics(self, metrics: Dict[str, float], fold_num: int):
        for key, val in metrics.items():
            metrics[key] = val / float(fold_num)

    def get_subconfig(self, prefix: str) -> Dict[str, any]:
        return {key[len(prefix)+1:]:value for key, value in self.config.items() if key.startswith(prefix + '.')}

    def create_feature(self, feature_name: str, dataset: DataSet, base_path: str, fold: DataPaths) -> Feature:
        return OneHotFeature(feature_name, dataset.labels(feature_name), **self.get_subconfig("input.features"))

    @abstractmethod
    def default_config(self) -> Dict[str, any]: raise NotImplementedError

    @abstractmethod
    def create_cross_validation_paths(self, base_path: str) -> List[DataPaths]: raise NotImplementedError

