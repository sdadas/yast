import copy
from collections import OrderedDict
from typing import List, Dict, Optional, TextIO
import csv, json

class DataSetFeature(object):

    def __init__(self, name: str, alphabet: Optional[List[str]]):
        self.name = name
        self.alphabet: Optional[List[str]] = self.__add_padding_symbol(alphabet)

    def __add_padding_symbol(self, alphabet: List[str]) -> Optional[List[str]]:
        if alphabet is None: return None
        if '<pad>' in alphabet: return alphabet
        else: return ['<pad>'] + alphabet

class DataSet(object):

    def __init__(self, data_path: Optional[str], meta_path: str, doc_data_path: Optional[str]=None, padding: int=None):
        """
        Creates new instance of DataSet and loads data.

        :param data_path: path to file containing words and word features
        :param doc_data_path: path to file containing sentence or document level features
        :param meta_path: path to meta.json file, containing feature metadata
        :param padding: padding length
        """
        self.data: List[List[Dict[str, str]]] = self._load_data(data_path) if data_path is not None else []
        self.docdata: List[Dict[str, str]] = self._load_docdata(doc_data_path) if doc_data_path is not None else None
        self.meta: Dict[str, DataSetFeature] = self._load_meta(meta_path)
        self._padding = padding
        if self.docdata is not None:
            assert len(self.data) == len(self.docdata), \
                f"document number mismatch, data({len(self.data)}) != docdata({len(self.docdata)})"

    def _load_data(self, path: str) -> List[List[Dict[str, str]]]:
        csv.register_dialect('space', delimiter=' ', quoting=csv.QUOTE_NONE)
        with open(path, 'r', newline='', encoding='utf-8') as input_file:
            reader = csv.reader(input_file, dialect='space')
            self._fieldnames = next(reader)
            self._maxlen = 0
            res: List[List[Dict[str, str]]] = []
            sentence: List[Dict[str, str]] = []
            for row in reader:
                if len(row) > 0:
                    row_dict = OrderedDict(zip(self._fieldnames, row))
                    sentence.append(row_dict)
                elif len(sentence) > 0:
                    self._maxlen = max(self._maxlen, len(sentence))
                    res.append(sentence)
                    sentence = []
            if len(sentence) > 0: res.append(sentence)
            return res

    def _load_docdata(self, path: str) -> List[Dict[str, str]]:
        csv.register_dialect('space', delimiter=' ', quoting=csv.QUOTE_NONE)
        with open(path, 'r', newline='', encoding='utf-8') as input_file:
            reader = csv.reader(input_file, dialect='space')
            self._doc_fieldnames = next(reader)
            res: List[Dict[str, str]] = []
            for row in reader:
                row_dict = OrderedDict(zip(self._doc_fieldnames, row))
                res.append(row_dict)
            return res

    def _load_meta(self, path: str) -> Dict[str, DataSetFeature]:
        with open(path, 'r', encoding='utf-8') as input_file:
            meta: Dict[str, List[str]] = json.load(input_file)
            res: Dict[str, DataSetFeature] = dict()
            for key in meta.keys():
                res[key] = DataSetFeature(key, meta.get(key, None))
        return res

    def __len__(self) -> int:
        return len(self.data)

    def copy(self):
        return copy.copy(self)

    def set_data(self, data: List[List[Dict[str, str]]]):
        self.data = data
        self._maxlen = max([len(row) for row in data]) if data else 0

    def set_docdata(self, docdata: List[Dict[str, str]]):
        self.docdata = docdata

    def sentence_length(self) -> int:
        return self._maxlen if self._padding is None else self._padding

    def labels(self, feature: str) -> List[str]:
        return self.meta[feature].alphabet

    def values(self, feature: str) -> List[List[str]]:
        return [[row[feature] for row in sent] for sent in self.data]

    def docvalues(self, feature: str) -> List[str]:
        return [row[feature] for row in self.docdata]

    def column(self, feature: str) -> DataSetFeature:
        return self.meta[feature]

    def add_feature(self, name: str, values: List[List[str]]):
        self._fieldnames.append(name)
        if name not in self.meta.keys():
            self.meta[name] = DataSetFeature(name, None)
        for sent_idx, sent in enumerate(self.data):
            for word_idx, word in enumerate(sent):
                word[name] = values[sent_idx][word_idx]

    def write(self, output_path: str):
        with open(output_path, 'wt', encoding='utf-8') as output_file:
            output_file.write(" ".join(self._fieldnames))
            output_file.write("\n")
            for sent in self.data:
                for word in sent:
                    self.__write_word(word, output_file)
                output_file.write("\n")

    def __write_word(self, word: Dict[str, str], output_file: TextIO):
        line = [word[fname] for fname in self._fieldnames]
        output_file.write(" ".join(line))
        output_file.write("\n")

    @staticmethod
    def empty(meta_path: str, fieldnames: List[str], padding: int):
        res = DataSet(None, meta_path, padding=padding)
        res._fieldnames = fieldnames
        return res