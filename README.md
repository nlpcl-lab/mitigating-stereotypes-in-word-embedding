Mitigating stereotypes in Word embedding through sentiment modulation
================================================
We provide the code and data for the following paper: Mitigating Stereotypes in Word Embedding through Sentiment Modulation by Huije Lee, Jin-Woo Chung, and Jong C. Park. Korea Software Congress 2018 (KSC 2018), Pyeongchang (Korea), December 2018. This repository provides a model that mitigates stereotypes in word embedding through sentiment modulation. 

### Requirements
- Python >= 3.5
- Tensorflow >= 1.10.0
- scikit-learn >=	0.19.1
- keras == 0.3.3

## Getting a word embedding with mitigating stereotypes
### Prerequistes
* Original word embedding for mitigating stereotype (e.g. [word2vec google-news vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors), [glove.6b.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip))
* Opinion Lexicon
* Wiki_vocabs_annotated for training a classifier to distinguish a human entity from others.
  - Note that all external resources should be located at ```source/```

### Commands
- ```python mitigating_stereotypes.py``` generates a word embedding with mitigating stereotypes
- ```python show_statistics.py``` shows the performance results of the original embedding and the mitigated embedding.

### Examples


### Results
- Intrinsic evaluation
- UCI evaluation

### References

Please cite following paper if using this code for learning word representations.

```
@InProceedings{lee2018mitigating,
  title={Mitigating Stereotypes in Word Embedding through Sentiment Modulation},
  author={Huije Lee, Jin-Woo Chung, and Jong C. Park},
  month={December},
  year={2018},
  publisher={The Korean Institute of Information Scientists and Engineers},
  pages={545--547},
}
```
- Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA.
- Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing and Comparing Opinions on the Web." Proceedings of the 14th International World Wide Web conference (WWW-2005), May 10-14, 2005, Chiba, Japan.
- William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from
Unlabeled Corpora. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), Nov 1-5, 2016, Austin, Texas, USA.
