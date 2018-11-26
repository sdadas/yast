import base64
import os
import unittest
from typing import List, Any, Iterable

import numpy as np
from keras import Input
from keras.engine import Layer
from keras.layers import Embedding
from pexpect import popen_spawn
from pexpect.popen_spawn import PopenSpawn

from dataset import DataSet
from feature.base import Feature
from utils.files import ProjectPath


class FSTFeature(Feature):

    lexicon: PopenSpawn = None

    def __init__(self, input_name: str, lexicon_name: str, alphabet: List[str], fst_path: ProjectPath,
                 trainable: bool=True, random_weights: bool=True, otag: str="other"):
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
        shape = [len(dataset), dataset.sentence_length()]
        res: np.ndarray = np.zeros(shape, dtype='int32')
        for sent_idx, sent in enumerate(dataset.data):
            words: Iterable[str] = [word[self.name()] for word in sent]
            indices: Iterable[str] = self.parse_sentence(words)
            for word_idx, label_idx in enumerate(indices):
                if word_idx >= dataset.sentence_length(): break
                idx = int(label_idx) + 1 if label_idx else self.__otag_idx
                res[sent_idx, word_idx] = idx
        return res

    def parse_sentence(self, words: Iterable[str], return_labels=False) -> Iterable[str]:
        encoded: bytes = base64.standard_b64encode("\n".join(words).encode("utf-8"))
        FSTFeature.lexicon.sendline("--parse {} {}".format(self.__lexicon_name, encoded.decode("utf-8")))
        output = FSTFeature.lexicon.readline().strip()
        print(output)
        res: bytes = base64.standard_b64decode(output.encode())
        indices: Iterable[str] = res.decode("utf-8").split("\n")
        if not return_labels: return indices
        else: return [self.__alphabet[int(label_idx) + 1] if label_idx else self.__otag for label_idx in indices]

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


class FSTFeatureTest(unittest.TestCase):

    def test_simple_fst(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        fst_dir = os.path.join(dir, "..", "examples", "fst")
        meta_file = os.path.join(fst_dir, "meta.json")
        dataset: DataSet = DataSet(path=None, meta_path=meta_file, padding=80)
        os.environ["FST_PATH"] = fst_dir
        feature: FSTFeature = FSTFeature("value", "food", dataset.labels("food"), ProjectPath("FST_PATH", "food.fst"))
        labels: Iterable[str] = feature.parse_sentence(["pizza", "kiełbasa", "miso", "gazpacho", "korma"], True)
        self.assertEquals(labels, ["italian", "polish", "japanese", "spanish", "other"])
