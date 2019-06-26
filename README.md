Mitigating stereotypes in Word embedding through sentiment modulation
================================================
We provide the code and data for the following paper: Mitigating Stereotypes in Word Embedding through Sentiment Modulation by Huije Lee, Jin-Woo Chung, and Jong C. Park. Korea Software Congress 2018 (KSC 2018), Pyeongchang (Korea), December 2018. This repository provides a model that mitigates stereotypes in word embedding through sentiment modulation. 

### Requirements
- Python >= 3.5
- Tensorflow >= 1.10.0
- scikit-learn >=	0.19.1
- keras == 0.3.3

## Get a word embedding with mitigating stereotypes
### Prerequistes
* Original word embedding for mitigating stereotype (e.g. [word2vec google-news vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) with binary=False, [glove.6b.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip))
* Opinion Lexicon
  - Note that all external resources should be located at ```source/```

### Code
- Prepare an original word embedding
- ```python mitigating_stereotypes.py``` generate a word embedding with mitigating stereotypes


### Examples


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
