import base64
import logging
import os
import unittest
from typing import List, Any, Iterable, Dict

import numpy as np
from keras import Input
from keras.engine import Layer
from keras.layers import Embedding
from pexpect import popen_spawn
from pexpect.popen_spawn import PopenSpawn

from dataset import DataSet
from feature.base import Feature
from utils.files import ProjectPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FSTFeature(Feature):

    lexicon: PopenSpawn = None
    batch_size: int = 64
    sentence_separator: str = "==!END!=="

    def __init__(self, input_name: str, lexicon_name: str, alphabet: List[str], fst_path: ProjectPath,
                 trainable: bool=True, random_weights: bool=True, otag: str="other", to_lower: str="no"):
        assert to_lower in ("first", "all", "no")
        self.__input_name = input_name
        self.__lexicon_name = lexicon_name
        self.__path = fst_path
        self.__alphabet = {idx: val for idx, val in enumerate(alphabet)}
        self.__otag: str = otag
        self.__otag_idx: int = alphabet.index(self.__otag)
        self.__size = len(alphabet)
        self.__random = random_weights
        self.__trainable = trainable
        self.__weights: np.ndarray = self.__init_weights(random_weights)
        self.__to_lower: str = to_lower
        self.run_fstlexicon()
        self.load_fstlexicon()

    def input(self):
        return Input(shape=(None,), dtype='int32', name=self.__lexicon_name + '_fstlexicon_input')

    def __init_weights(self, random_weights: bool=False) -> np.ndarray:
        if random_weights: return np.random.uniform(-0.5, 0.5, size=(self.__size, self.__size))
        else: return np.identity(self.__size, dtype='float32')

    def model(self, input: Any) -> Layer:
        w = [self.__weights]
        size = self.__size
        name = self.__lexicon_name + '_embedding'
        return Embedding(output_dim=size, input_dim=size, weights=w, trainable=self.__trainable, name=name)(input)

    def transform(self, dataset: DataSet):
        logger.info("Transforming %d sentences with '%s' FST", len(dataset), self.__lexicon_name)
        shape = [len(dataset), dataset.sentence_length()]
        res: np.ndarray = np.zeros(shape, dtype='int32')
        batch: List[List[str]] = []
        sent_idx_start = 0
        for sent_idx, sent in enumerate(dataset.data):
            words: List[str] = [word[self.__input_name] for word in sent]
            if self.__to_lower == "all": words = [word.lower() for word in words]
            elif self.__to_lower == "first": words[0] = words[0].lower()
            batch.append(words)
            if len(batch) >= self.batch_size or sent_idx >= (len(dataset) - 1):
                self.__transform_batch(sent_idx_start, res, dataset, batch)
                batch = []
                sent_idx_start = sent_idx + 1
        return res

    def __transform_batch(self, sent_idx_start: int, res: np.ndarray, dataset: DataSet, batch: List[List[str]]):
        indices: List[List[str]] = self.parse_batch(batch)
        assert len(batch) == len(indices), f"{len(batch)} != {len(indices)} for batch ({sent_idx_start}+)"
        idx = 0
        while idx < len(indices):
            sent_indices: List[str] = indices[idx]
            for word_idx, label_idx in enumerate(sent_indices):
                if word_idx >= dataset.sentence_length(): break
                output_idx = int(label_idx) + 1 if label_idx else self.__otag_idx
                res[sent_idx_start + idx, word_idx] = output_idx
            idx += 1

    def parse_batch(self, batch: List[List[str]], return_labels=False) -> List[List[str]]:
        sentences = ["\n".join(words) for words in batch]
        encoded: bytes = base64.standard_b64encode(self.sentence_separator.join(sentences).encode("utf-8"))
        FSTFeature.lexicon.sendline("--parse {} {}".format(self.__lexicon_name, encoded.decode("utf-8")))
        output = FSTFeature.lexicon.readline().strip()
        res: bytes = base64.standard_b64decode(output.encode())
        sent_indices: List[str] = res.decode("utf-8").split(self.sentence_separator)
        indices: List[List[str]] = [sent.split("\n") for sent in sent_indices]
        if not return_labels: return indices
        else: return [self.indices2words(sent) for sent in indices]

    def parse_sentence(self, words: Iterable[str], return_labels=False) -> Iterable[str]:
        encoded: bytes = base64.standard_b64encode("\n".join(words).encode("utf-8"))
        FSTFeature.lexicon.sendline("--parse {} {}".format(self.__lexicon_name, encoded.decode("utf-8")))
        output = FSTFeature.lexicon.readline().strip()
        res: bytes = base64.standard_b64decode(output.encode())
        indices: List[str] = res.decode("utf-8").split("\n")
        if not return_labels: return indices
        else: return self.indices2words(indices)

    def indices2words(self, sentence: List[str]) -> List[str]:
        return [self.__alphabet[int(label_idx) + 1] if label_idx else self.__otag for label_idx in sentence]

    def alphabet(self) -> Dict[int, str]:
        return self.__alphabet

    def run_fstlexicon(self):
        if FSTFeature.lexicon is not None: return
        dir = os.path.dirname(os.path.realpath(__file__))
        jar_file = os.path.join(dir, 'fstlexicon.jar').replace('\\', '/')
        process: PopenSpawn = popen_spawn.PopenSpawn('java -jar %s' % (jar_file,), encoding='utf-8')
        FSTFeature.lexicon: PopenSpawn = process

    def load_fstlexicon(self):
        path: str = self.__path.get().replace('\\', '/')
        FSTFeature.lexicon.sendline("--load {} {}".format(self.__lexicon_name, path))

    def name(self) -> str:
        return self.__lexicon_name

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.run_fstlexicon()
        self.load_fstlexicon()


class FSTFeatureTest(unittest.TestCase):

    def test_simple_fst(self):
        example_dir = os.path.dirname(os.path.realpath(__file__))
        fst_dir = os.path.join(example_dir, "..", "examples", "fst")
        meta_file = os.path.join(fst_dir, "meta.json")
        dataset: DataSet = DataSet(data_path=None, meta_path=meta_file, padding=80)
        os.environ["FST_PATH"] = fst_dir
        feature: FSTFeature = FSTFeature("value", "food", dataset.labels("food"), ProjectPath("FST_PATH", "food.fst"))

        batch = [["pizza", "kiełbasa", "miso", "gazpacho", "korma"], ["gnocchi", "pierogi", "xiaomi"]]
        answers = [["italian", "polish", "japanese", "spanish", "other"], ["italian", "polish", "other"]]
        labels: List[List[str]] = feature.parse_batch(batch, True)
        self.assertEquals(labels, answers)
