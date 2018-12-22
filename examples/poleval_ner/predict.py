import argparse
import json
import logging
import os
from collections import OrderedDict
from typing import List, Dict, Tuple

from dataset import DataSet
from model import TaggingModel
from utils.files import ProjectPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    default_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=default_path)
    return parser.parse_args()


def predict(model_path: str, dataset: DataSet) -> List[List[str]]:
    model: TaggingModel = TaggingModel.load(model_path)
    result: List[List[str]] = model.predict(dataset, string_labels=True, verbose=1)
    return result


class PolevalJsonFormat(object):

    def __init__(self, data: DataSet, ids_path: str, correct_labels: bool=True):
        self.data: DataSet = data
        self.label_map: Dict[str, str] = self.create_label_map()
        self.correct_labels = correct_labels
        with open(ids_path, "r", encoding="utf-8") as ids_file:
            self.ids: List[str] = [line.strip() for line in ids_file]
        self.__preprocess_data()

    def create_label_map(self) -> Dict[str, str]:
        return {
            "addName": "persName_addName",
            "forename": "persName_forename",
            "surname": "persName_surname",
            "district": "placeName_district",
            "settlement": "placeName_settlement",
            "region": "placeName_region",
            "country": "placeName_country",
            "bloc": "placeName_bloc"
        }

    def write(self, output_path: str):
        res: List[Dict] = []
        current_id: str = self.ids[0]
        current_answers: List[Tuple] = []
        for sent_idx, sentence in enumerate(self.data.data):
            type_spans = self.__get_schemed_spans(sentence, "type")
            subtype_spans = self.__get_unschemed_spans(sentence, "subtype")
            spans = type_spans + subtype_spans
            answers = self.__get_answers(sentence, spans)
            sent_id = self.ids[int(sentence[0]["paragraph"])]
            if sent_id != current_id:
                self.__add_answers(current_answers, current_id, res)
                current_id = sent_id
                current_answers = answers
            else:
                current_answers.extend(answers)
        self.__add_answers(current_answers, current_id, res)
        self.__write_results(output_path, res)

    def __write_results(self, output_path: str, res: List[Dict]):
        with open(output_path, "w", encoding="utf-8") as output_file:
            if output_path.lower().endswith(".txt"):
                for row in res:
                    output_file.write(row["id"])
                    output_file.write("\n")
                    output_file.write(row["answers"])
                    output_file.write("\n\n")
            else: json.dump(res, output_file, sort_keys=False, indent=2)

    def __preprocess_data(self):
        for sentence in self.data.data:
            for word in sentence:
                self.__preprocess_word(word)

    def __preprocess_word(self, word: Dict[str, str]):
        type_label = word["type"]
        subtype_label = word["subtype"]
        if type_label == "O" or type_label == "<pad>":
            word["subtype"] = "O"
        else:
            type_label = type_label[2:]
            if type_label == subtype_label: word["subtype"] = "O"
            else: word["subtype"] = self.label_map.get(subtype_label, subtype_label)
            if type_label == "placeName" and word["subtype"] == "O": word["subtype"] = "placeName_country"

    def __add_answers(self, current_answers: List[Tuple], current_id: str, res: List[Dict]):
        current_answers = sorted(current_answers, key=lambda val: (int(val[1]), int(val[2])))
        output_answers = []
        for answer in current_answers:
            label = answer[0]
            if label == "placeName": continue
            output_answers.append(f"{label} {answer[1]} {answer[2]}\t{answer[3]}")
        output_answers = list(OrderedDict.fromkeys(output_answers))
        res.append({"id": current_id, "text": "", "answers": "\n".join(output_answers)})

    def __get_answers(self, sentence: List[Dict[str, str]], spans: List[Tuple]) -> List[Tuple]:
        res: List[Tuple] = []
        for span in spans:
            label = span[2]
            idx_from = sentence[span[0]]["idx_from"]
            idx_to = sentence[span[1]]["idx_to"]
            value = " ".join([sentence[span[0] + idx]["value"] for idx in range(span[1] - span[0] + 1)])
            res.append((label, idx_from, idx_to, value))
        return res

    def __get_schemed_spans(self, sentence: List[Dict[str, str]], feature_name: str) -> List[Tuple]:
        start = None
        current_label = None
        spans = []
        for pos, word in enumerate(sentence):
            label = word[feature_name]
            if label.startswith("B-") or label.startswith("S-") or label == "O" or label == "<pad>":
                if start is not None: spans.append((start, pos-1, current_label))
            if label.startswith("E-") or label.startswith("I-"):
                if current_label != label[2:]:
                    if self.correct_labels:
                        label = self.__label_correction(current_label, label[2:], pos, sentence, feature_name)
                    else: raise AssertionError(f"incorrect label ({current_label}) -> ({word['value']} {label})")
            if label.startswith("B-") or label.startswith("S-"):
                start = pos
                current_label = label[2:]
            if label.startswith("E-") or label.startswith("S-"):
                spans.append((start, pos, current_label))
                start = None
                current_label = None
            if label == "O" or label == "<pad>":
                start = None
                current_label = None
        if start is not None: spans.append((start, len(sentence) - 1, current_label))
        return spans

    def __label_correction(self, previous: str, current: str, pos: int, sentence: List[Dict[str, str]], fname: str):
        next = sentence[pos + 1][fname] if len(sentence) > (pos + 1) else None
        next = None if next is None or next == "O" or next == "<pad>" else next[2:]
        if next is None: return "O"
        elif next == current: return "B-" + next
        elif previous == next: return "I-" + previous
        else: return "O"

    def __get_unschemed_spans(self, sentence: List[Dict[str, str]], feature_name: str) -> List[Tuple]:
        start = None
        current_label = None
        spans = []
        for pos, word in enumerate(sentence):
            label = word[feature_name]
            if label == "O" or label == "<pad>" or (start is not None and current_label != label):
                if start is not None: spans.append((start, pos - 1, current_label))
                start = None
                current_label = None
            if label != "O" and label != "<pad>" and current_label != label:
                start = pos
                current_label = label
        if start is not None: spans.append((start, len(sentence) - 1, current_label))
        return spans


if __name__ == '__main__':
    args = parse_args()
    os.environ["NER_PATH"] = args.data_path
    path: ProjectPath = ProjectPath("NER_PATH")
    model_path = path.join("model").get()
    submodel_path = path.join("submodel").get()
    meta_path = path.join("meta.json").get()

    logging.info("Loading dataset")
    dataset: DataSet = DataSet(path.join("poleval.txt").get(), meta_path, padding=80)
    logging.info("Predicting by type model")
    type_prediction: List[List[str]] = predict(model_path, dataset)
    dataset.add_feature("type", type_prediction)
    logging.info("Predicting by subtype model")
    subtype_prediction: List[List[str]] = predict(submodel_path, dataset)
    dataset.add_feature("subtype", subtype_prediction)
    logging.info("Writing results")
    dataset.write(path.join("results.txt").get())
    output = PolevalJsonFormat(dataset, path.join("poleval.ids.txt").get())
    output.write(path.join("results.json").get())
