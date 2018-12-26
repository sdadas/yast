import os
from typing import List, Tuple

from dataset import DataSet
from feature.base import Feature
from feature.casing import CasingFeature
from feature.embeddings import EmbeddingFeature
from model import TaggingModel, TaggingPrediction, ClassificationPrediction, ModelParams
from utils.files import ProjectPath


def create_features(base_path: ProjectPath) -> List[Feature]:
    return [
        EmbeddingFeature("value", base_path.join("glove_200d.txt"), text_format=True, trainable=True),
        CasingFeature("value")
    ]

if __name__ == '__main__':
    os.environ["ATIS_PATH"] = os.path.dirname(os.path.realpath(__file__))
    path: ProjectPath = ProjectPath("ATIS_PATH")
    meta = path.join("meta.json").get()
    train_paths = [path.join("train.sequences.txt").get(), path.join("train.labels.txt").get()]
    train: DataSet = DataSet(train_paths[0], meta, train_paths[1], padding=30)
    test_paths = [path.join("test.sequences.txt").get(), path.join("test.labels.txt").get()]
    test: DataSet = DataSet(test_paths[0], meta, test_paths[1], padding=30)

    features: List[Feature] = create_features(path)
    params: ModelParams = ModelParams(lstm_layers=1, lstm_size=200, learning_rate=0.002)
    model = TaggingModel(features, train.column("label"), train.column("doclabel"))
    model.train(train, epochs=50)
    TaggingModel.save(model, path.join("model").get())
    pred: Tuple[TaggingPrediction, ClassificationPrediction] = model.test(test)
    pred[0].evaluate()
    pred[1].evaluate()
