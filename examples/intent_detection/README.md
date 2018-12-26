## Intent detection & slot filling example

This example shows how to create a model that can jointly tag and classify sentences, on the example of ATIS intent detection and slot filling task. See `example.py` for details.

The example uses GloVe 200 dimensional embedding vectors with all the words removed except for those present in the ATIS data set (to reduce the size of embeddings).