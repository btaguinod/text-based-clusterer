import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import math

nltk.download('stopwords')
nltk.download('punkt')


class NLPObject:
    """Object Container with vector for similarity comparisons.

    Attributes:
        object: Contained object.
        tokens:
        vector: Numpy vector representation.
    """

    def __init__(self, object_arg: object, attributes: list[str]):
        """Initialize NLPObject

        Args:
            object_arg: Object for container.
            attributes: List of strings corresponding to object text
                attributes.
        """
        self.object = object_arg
        self.tokens = self.preprocess_text(attributes)
        self.vector = None

    def set_tf_vector(self, index: list[str]) -> list[str]:
        """Calculate and store term frequency vector.

        Args:
            index: Word index.

        Returns:
            Updated word index.
        """
        new_index = index.copy()
        unique_words = set(self.tokens)

        text_freq_dict = {word: 0 for word in unique_words}
        for word in self.tokens:
            text_freq_dict[word] += 1

        if new_index is None:
            new_index = list(unique_words)
        else:
            addition = [word for word in unique_words if word not in new_index]
            new_index += addition

        tf = []
        for word in new_index:
            tf.append(text_freq_dict[word] if word in unique_words else 0)

        self.vector = np.array(tf) / len(self.tokens)
        return new_index

    def pad_tf_vector(self, padding_len: int):
        """Pad the term frequency vector with zeros.

        Args:
            padding_len: Length vector will be extended by.
        """
        self.vector = np.pad(self.vector, (0, padding_len))

    def preprocess_text(self, attributes: list[str]) -> list[str]:
        """Turn text into tokens without stopwords and punctuation.

        Args:
            attributes: List of strings corresponding to object text
                attributes.

        Returns:
            List of tokens.
        """
        text = ''
        for attribute in attributes:
            text += getattr(self.object, attribute) + ' '
        text = text.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if token.lower not in stop_words:
                filtered_tokens.append(token)
        return filtered_tokens


class Cluster:
    """Cluster of NLPObject objects.

    Attributes:
        nlp_objects: Stored NLPObject objects
        self.vector: Average of term frequency vectors in NLPObject objects.
    """

    def __init__(self, nlp_objects: list[NLPObject]):
        self.nlp_objects = nlp_objects
        self.vector = None

    def set_tf_vector(self, index: list[str]):
        """Calculate and store average term frequency vector.

        Args:
            index: Word index.
        """
        new_index = index
        self.vector = np.zeros(len(new_index))
        for nlp_object in self.nlp_objects:
            self.vector += nlp_object.vector
        self.vector /= len(self.nlp_objects)

    def pad_tf_vector(self, padding_len: int):
        """Pad the term frequency vector with zeros.

        Args:
            padding_len: Length vector will be extended by.
        """
        self.vector = np.pad(self.vector, (0, padding_len))

    def get_list(self) -> list[object]:
        """Gets list representation.

        Returns:
            List representation.
        """
        objects = []
        for nlp_object in self.nlp_objects:
            objects.append(nlp_object.object)
        return objects


class Clusterer:
    """Clusters objects.

    Attributes:
        cluster_threshold: Cosine similarity from 0 to 1 for clustering
            strictness.
        clusters: Stored Cluster objects.
        index: Word index.
    """

    def __init__(self, cluster_threshold: float = 0.3):
        self.cluster_threshold = cluster_threshold
        self.clusters = []
        self.index = []

    def add_clusters(self, cluster_lists: list[list[object]], attributes: list[str]):
        """Add existing clusters.

        Args:
            cluster_lists: Clustered lists of objects.
            attributes: List of strings corresponding to object text
                attributes.
        """
        for cluster_list in cluster_lists:
            nlp_objects = []
            for cluster_object in cluster_list:
                nlp_objects.append(NLPObject(cluster_object, attributes))
            self.clusters.append(Cluster(nlp_objects))

        for cluster in self.clusters:
            for nlp_object in cluster.nlp_objects:
                self.index = nlp_object.set_tf_vector(self.index)
            index_len = len(self.index)
            for nlp_object in cluster.nlp_objects:
                padding_len = index_len - len(nlp_object.vector)
                nlp_object.pad_tf_vector(padding_len)
            cluster.set_tf_vector(self.index)

    def add_objects(self, objects: list[object], attributes: list[string]):
        """Add objects to clusters.

        Args:
            objects: Objects to cluster.
            attributes: List of strings corresponding to attributes.
        """
        nlp_objects = [NLPObject(object_item, attributes) for object_item in objects]
        for nlp_object in nlp_objects:
            self.index = nlp_object.set_tf_vector(self.index)

        index_len = len(self.index)
        for nlp_object in nlp_objects:
            padding_len = index_len - len(nlp_object.vector)
            nlp_object.pad_tf_vector(padding_len)
        for cluster in self.clusters:
            padding_len = index_len - len(cluster.vector)
            for cluster_nlp_object in cluster.nlp_objects:
                cluster_nlp_object.pad_tf_vector(padding_len)
            cluster.pad_tf_vector(padding_len)

        inv_doc_freq = self.get_inv_doc_freq(nlp_objects)
        for nlp_object in nlp_objects:
            closest_dist = 0
            closest_cluster = None

            for cluster in self.clusters.copy():
                new_dist = vector_similarity(
                    np.multiply(nlp_object.vector, inv_doc_freq),
                    np.multiply(cluster.vector, inv_doc_freq))

                if new_dist > self.cluster_threshold and new_dist > closest_dist:
                    closest_dist = new_dist
                    closest_cluster = cluster

            if closest_cluster is None:
                new_cluster = Cluster([nlp_object])
                self.clusters.append(new_cluster)
                new_cluster.set_tf_vector(self.index)
            else:
                closest_cluster.nlp_objects.append(nlp_object)
                closest_cluster.set_tf_vector(self.index)

    def get_clusters(self) -> list[list[object]]:
        """Get clustered lists of objects.

        Returns:
            clustered lists of objects.
        """
        return [cluster.get_list() for cluster in self.clusters]

    def get_inv_doc_freq(self, nlp_objects: list[NLPObject]) -> np.ndarray:
        """Calculate inverse document frequency.

        Args:
            nlp_objects: Extra NLPObjects to use in calculations.
        """
        vector_len = len(self.index)

        doc_freq = np.zeros(vector_len)
        for nlp_object in nlp_objects:
            doc_freq += np.ceil(nlp_object.vector)
        for cluster in self.clusters:
            for nlp_object in cluster.nlp_objects:
                doc_freq += np.ceil(nlp_object.vector)

        doc_count = len(nlp_objects)
        for cluster in self.clusters:
            doc_count += len(cluster.nlp_objects)

        inv_doc_freq = np.zeros(vector_len)
        for i in range(len(doc_freq)):
            num = doc_freq[i]
            if num != 0:
                num = math.log((1 / num) * doc_count)
                inv_doc_freq[i] = num

        return inv_doc_freq


def vector_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between vectors.

    Args:
       a: First numpy vector.
       b: Second numpy vector.

    Returns:
        float: Cosine similarity.
    """
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
