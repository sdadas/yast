## PolEval 2018 NER Example

This example is an implementation of the method from our article "Combining neural and knowledge-based approaches to Named Entity Recognition in Polish”, 2018; [arXiv:1811.10418](https://arxiv.org/abs/1811.10418).

#### 1. Downloading the data

Download and extract [the data needed to run this example](https://hkbaom-my.sharepoint.com/:u:/g/personal/pro12151_my365_site/EeN-fn7HhUlDvI3HuKtJMbwB8iuvppyrDmezJZzhoKp1tg?e=ZTJnth). The archive contains pre-trained ELMo embeddings for Polish, five lexicons in Lucene FST format, pre-trained models, preprocessed training (NKJP) and evaluation (PolEval) data sets.
In the case of NKJP data set, preprocessing included running Wikipedia Miner entity linking algorithm on the data. For PolEval, the data has been tokenized, lemmatized and then processed in Wikipedia Miner.
Results of entity linking are stored in `wikipedia` column in both data sets. 

#### 2. Training

If you want to use pre-trained models provided by us, you can skip this step and go directly to `Making predictions`.

This example requires two models to be trained: one for predicting main entity categories and one for predicting sub-categories, therefore the training script needs to be executed twice.
For category model, run `train.py` without arguments, for sub-category model, run `train.py --submodel`.
The models use all the lexicons and entities extracted from Wikipedia by default. 
One can exclude external features from the model by adding `--no-lexicons` or `--no-wikipedia` options  respectively.

#### 3. Making predictions

Running the `predict.py` script generates two files: `results.txt` which contains the predictions of both model and submodel in plain text format, and `results.json` containing the same predictions in PolEval compatibile json format.

#### 4. Evaluation

In order to evaluate the models on the PolEval data set, you need to download two files provided by the competition organizers: [golden answers](http://mozart.ipipan.waw.pl/~axw/poleval2018/POLEVAL-NER_GOLD.json) and the [official evaluation script](http://mozart.ipipan.waw.pl/~axw/poleval2018/poleval_ner_test.py).
After that, run the evaluation script on the predictions generated in the previous step:

```text
poleval_ner_test.py -g POLEVAL-NER_GOLD.json -u data/results.json
```

Models trained for 10 epochs should reach a final score of about 90. Evaluation results for the pre-trained models compared to the three best submissions from the PolEval competition are shown in the table below. 

| Model              | Final score | Exact score | Overlap score |
|--------------------|:-----------:|:-----------:|:-------------:|
| Liner2             |     81.0    |     77.8    |      81.8     |
| PolDeepNer         |     85.1    |     82.2    |      85.9     |
| Per group LSTM-CRF |     86.6    |     82.6    |      87.7     |
| This code (yast)   |     90.3    |     87.0    |      91.1       |