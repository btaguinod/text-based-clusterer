import unittest

from src.text_based_clusterer.clusterer import Clusterer


class StringContainer:
    def __init__(self, str_1: str, str_2: str):
        self.str_1 = str_1
        self.str_2 = str_2

    def __repr__(self):
        return '"' + self.str_1 + ' ' + self.str_2 + '"'


class ClustererTest(unittest.TestCase):
    CLUSTER_THRESHOLD = 0.3

    def test_single_attribute(self):
        print('Testing clustering using one attribute:')

        string_container_1 = StringContainer('dog', 'cat')
        string_container_2 = StringContainer('cat', 'dog')
        string_container_3 = StringContainer('goat', 'sheep')
        string_container_4 = StringContainer('goat', 'goat')

        string_containers = [string_container_1, string_container_2,
                             string_container_3, string_container_4]

        clusterer = Clusterer(self.CLUSTER_THRESHOLD)

        clusterer.add_objects(string_containers, ['str_1'])

        print(clusterer.get_clusters())

    def test_multiple_attributes(self):
        print('Testing clustering using two attributes:')

        string_container_1 = StringContainer('dog', 'cat')
        string_container_2 = StringContainer('cat', 'dog')
        string_container_3 = StringContainer('goat', 'sheep')
        string_container_4 = StringContainer('goat', 'goat')

        string_containers = [string_container_1, string_container_2,
                             string_container_3, string_container_4]

        clusterer = Clusterer(self.CLUSTER_THRESHOLD)

        clusterer.add_objects(string_containers, ['str_1', 'str_2'])

        print(clusterer.get_clusters())


if __name__ == '__main__':
    unittest.main()
