import os
import unittest

class ProjectPath(object):

    def __init__(self, path_env: str, relative_path: str="."):
        self.__path_env = path_env
        self.__relative_path = relative_path

    def join(self, *paths):
        return ProjectPath(self.__path_env, os.path.join(self.__relative_path, *paths))

    def get(self) -> str:
        return os.path.join(os.environ[self.__path_env], self.__relative_path)


class ProjectPathTests(unittest.TestCase):

    def test_path(self):
        os.environ["TEST"] = os.path.dirname(os.path.realpath(__file__))
        path: ProjectPath = ProjectPath("TEST")
        self.assertTrue(os.path.exists(path.join("files.py").get()))