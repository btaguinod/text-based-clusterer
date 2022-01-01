Text Based Clusterer
======

Allows you to cluster python objects based on chosen string attributes.

## Installation

To install, use `pip install text-based-clusterer`.

## Usage

### Instantiation

You can import the Clusterer class with the line below.
``` python
    from text_based_clusterer import Clusterer
```
To use the clusterer, first create a python class with a string attribute or use an existing python class with a string attribute.
``` python
    class TestClass:
        __init__(self, string1: str, string2: str):
            self.string1 = string1
            self.string2 = string2
```
Then instantiate the Cluster object. There is an optional parameter `cluster_threshold` that defines how textually similar objects should be to be clustered together. `cluster threshold` ranges from 0 (completely different) to 1 (exactly the same) and is set to a default of 0.3.
``` python
    clusterer = Clusterer(cluster_threshold = 0.3)
```

### Basic Clustering

Use the `add_objects` method to cluster objects. You will need to pass in a list of python objects as well as a list of strings representing the attributes used for clustering. You can retrieve a list of clusters using `get_objects`.
``` python
    test_class_1 = TestClass('dog', 'cat')
    test_class_2 = TestClass('cat', 'dog')

    test_classes = [test_class_1, test_class_2]


    single_string_clusterer = Clusterer()

    single_string_clusterer.add_objects(test_classes, ['string_1'])

    single_string_clusterer.get_objects() # returns [[test_class_1], [test_class_2]]


    multi_string_clusterer = Clusterer()

    clusterer.add_objects(test_classes, ['string_1', 'string_2'])

    multi_string_clusterer.get_objects() # returns [[test_class_1, test_class_2]] 
```

### Adding to Existing Clusters

Use the `add_clusters` method to add an initial set of clusters in the form of nested lists. Note that this method should only be used when there are no existing clusters within the `Cluster` object because calling `add_clusters` does not perform clustering.


``` python
    test_class_1 = TestClass('dog', 'cat')
    test_class_2 = TestClass('cat', 'dog')

    initial_cluster = [[test_class_1], [test_class_2]]

    clusterer = Clusterer()

    clusterer.add_clusters(initial_cluster, ['string_1'])

    clusterer.add_objects([test_class_1], ['string_1'])

    clusterer.get_objects() # returns [[test_class_1, test_class_1], [test_class_2]]
```
## Sources Used

This package implements a text based incremental clustering algorithm from the research paper [Incremental Clustering of News Reports](https://www.researchgate.net/publication/258028563_Incremental_Clustering_of_News_Report).
