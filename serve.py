import argparse
import os
from typing import List, Dict

from flask import Flask, request, jsonify

from dataset import DataSet
from model import TaggingModel


class PredictionHandler(object):

    def __init__(self, model_dir: str, fieldnames: List[str], padding: int=80):
        meta_path = os.path.join(model_dir, "meta.json")
        self.__model: TaggingModel = TaggingModel.load(model_dir)
        self.__data: DataSet = DataSet.empty(meta_path, fieldnames, padding)

    def predict(self, data: List[List[Dict[str, str]]]) -> Dict[str, any]:
        input_data = self.__data.copy()
        input_data.set_data(data)
        result = self.__model.predict(input_data, string_labels=True, verbose=0)
        if isinstance(result, tuple): return {"tags": result[0], "labels": result[1]}
        else: return {"tags": result}


def create_app():
    app = Flask(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--padding", type=int, default=80)
    parser.add_argument("--fieldnames", type=str, default="value")
    args = parser.parse_args()
    fieldnames = [fname.strip() for fname in args.fieldnames.split(sep=",")]
    server = PredictionHandler(args.model_dir, fieldnames, padding=args.padding)

    @app.route("/", methods=["POST"])
    def index() -> List[List[str]]:
        data: any = request.json
        sentences: List[List[Dict[str, str]]] = data.get("sentences")
        res = server.predict(sentences)
        return jsonify(res)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(threaded=False)
