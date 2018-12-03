import math
import unittest
from typing import List

from dataset import DataSet
from feature.base import OneHotFeature


class CasingFeature(OneHotFeature):

    def __init__(self, input_feature: str, trainable: bool=True, random_weights: bool=True):
        super().__init__(input_feature, CasingFeature.default_alphabet(), trainable=trainable, random_weights=random_weights)

    @staticmethod
    def default_alphabet() -> List[str]:
        return ["numeric", "mostly_numeric", "upper", "lower", "title", "contains_digit", "mixed", "other"]

    def transform(self, dataset: DataSet):
        return self._transform_by_func(dataset, lambda val: self.alphabet.get(self.get_casing(val)))

    def get_casing(self, word: str):
        first_upper = len(word) > 0 and word[0].isupper()
        numeric = 0
        lower = 0
        upper = 0
        other = 0
        for char in word:
            if char.isnumeric(): numeric += 1
            elif not char.isalnum(): other += 1
            elif char.isupper(): upper += 1
            elif char.islower(): lower += 1
            else: other += 1
        if numeric == len(word): return "numeric"
        elif numeric >= math.ceil(len(word) / 2.0): return "mostly_numeric"
        elif upper == len(word): return "upper"
        elif lower == len(word): return "lower"
        elif first_upper and upper == 1: return "title"
        elif numeric > 0: return "contains_digit"
        elif upper > 0 and lower > 0: return "mixed"
        else: return "other"


class CasingFeatureTest(unittest.TestCase):

    def test_casing_feature(self):
        feature: CasingFeature = CasingFeature("test")
        self.assertEqual(feature.get_casing("1234"), "numeric")
        self.assertEqual(feature.get_casing("Ala1234"), "mostly_numeric")
        self.assertEqual(feature.get_casing("ALA"), "upper")
        self.assertEqual(feature.get_casing("ala"), "lower")
        self.assertEqual(feature.get_casing("Ala"), "title")
        self.assertEqual(feature.get_casing("ala1"), "contains_digit")
        self.assertEqual(feature.get_casing("MixEdCase"), "mixed")
        self.assertEqual(feature.get_casing("!"), "other")
