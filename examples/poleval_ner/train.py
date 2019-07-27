import argparse
import os
from typing import List

from dataset import DataSet
from feature.base import OneHotFeature, Feature
from feature.casing import CasingFeature
from feature.chars import CharsFeature, CharCNNFeature
from feature.elmo import ELMoEmbeddingFeature
from feature.fstlexicon import FSTFeature
from model import TaggingModel, TaggingPrediction, ModelParams
from utils.files import ProjectPath


def parse_args() -> argparse.Namespace:
    default_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=default_path)
    parser.add_argument("--submodel", action="store_true")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--no-lexicons", action="store_true")
    return parser.parse_args()


def create_features(dataset: DataSet, base_path: ProjectPath, args: argparse.Namespace) -> List[Feature]:
    fst = lambda name: base_path.join("lexicons", name + ".fst")
    labels = lambda name: dataset.labels(name)
    features = [
        ELMoEmbeddingFeature("value", base_path.join("elmo")),
        CharCNNFeature("value", CharsFeature.default_alphabet()),
        CasingFeature("value")
    ]
    lexicons = [
        FSTFeature("base", "naming", labels("naming"), fst("names"), to_lower="all", otag="other"),
        FSTFeature("value", "polimorf", labels("polimorf"), fst("polimorf"), to_lower="first", otag="O"),
        FSTFeature("value", "gazetteer", labels("gazetteer"), fst("gazetteer"), to_lower="no", otag="other"),
        FSTFeature("value", "nelexicon", labels("nelexicon"), fst("nelexicon"), to_lower="no", otag="O"),
        FSTFeature("value", "extras", labels('extras'), fst("extras"), to_lower="no", otag="O")
    ]
    if not args.no_lexicons: features.extend(lexicons)
    if not args.no_wikipedia: features.append(OneHotFeature("wikipedia", labels("wikipedia")))
    if args.submodel: features.append(OneHotFeature("type", labels("type")))
    return features


if __name__ == '__main__':
    args = parse_args()
    os.environ["NER_PATH"] = args.data_path
    path: ProjectPath = ProjectPath("NER_PATH")
    meta_path = path.join("meta.json").get()
    train: DataSet = DataSet(path.join("nkjp.txt").get(), meta_path, padding=80)
    train, valid = train.train_test_split(0.95)
    features: List[Feature] = create_features(train, path, args)
    params: ModelParams = ModelParams(learning_rate=0.05)
    params.restore_best_weights()
    params.early_stopping(patience=10)
    params.sgd_with_restarts_scheduler(train, batch_size=32, max_lr=0.1, min_lr=0.0002, mult_factor=1.5)
    model = TaggingModel(features, train.column("subtype" if args.submodel else "type"), params=params)
    model.train(train, valid=valid, epochs=30)
    TaggingModel.save(model, path.join("submodel" if args.submodel else "model").get())
    pred: TaggingPrediction = model.test(train)
    pred.evaluate(ignore_tagging_scheme=args.submodel)
